"""polylogue-y964 / polylogue-4c27: the `delegations` view composes a
parent-dispatched subagent attempt from the PARENT's own dispatch actions
(`actions` rows, semantic_type='subagent'), corroborated against resolved
children via canonical `session_links` (child in `src_session_id`, parent in
`resolved_dst_session_id` -- see resolve_session_links_for_session). The
prior shipped view aliased these backwards; these fixtures use the canonical
direction throughout and would fail against that reversed view. Model
identity is separated into dispatch-turn / requested / child-observed /
session-dominant-fallback columns rather than one "orchestrator model"."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.topology.edge import TopologyEdgeRecord, TopologyEdgeStatus
from polylogue.core.enums import LinkType as TopologyEdgeType
from polylogue.core.enums import Origin
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.queries.session_links import (
    resolve_session_links_for_session,
    upsert_session_links,
)
from polylogue.types import SessionId

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


def _insert_message(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    native_id: str,
    position: int,
    model_name: str | None = None,
) -> str:
    conn.execute(
        """
        INSERT INTO messages (
            session_id, native_id, position, role, message_type, model_name, content_hash, occurred_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, native_id, position, "assistant", "message", model_name, _HASH, 1_767_225_600_000 + position),
    )
    return str(
        conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ? AND native_id = ?", (session_id, native_id)
        ).fetchone()["message_id"]
    )


def _insert_dispatch_action(
    conn: sqlite3.Connection,
    *,
    message_id: str,
    session_id: str,
    position: int,
    tool_id: str,
    tool_input: str = "{}",
    result_text: str | None = "done",
    result_is_error: int | None = 0,
    result_exit_code: int | None = 0,
) -> None:
    conn.execute(
        """
        INSERT INTO blocks (
            message_id, session_id, position, block_type, tool_name, tool_id, semantic_type, tool_input
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (message_id, session_id, position, "tool_use", "Task", tool_id, "subagent", tool_input),
    )
    if result_text is not None:
        conn.execute(
            """
            INSERT INTO blocks (
                message_id, session_id, position, block_type, text, tool_id,
                tool_result_is_error, tool_result_exit_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                session_id,
                position + 1,
                "tool_result",
                result_text,
                tool_id,
                result_is_error,
                result_exit_code,
            ),
        )


def _insert_session_profile(conn: sqlite3.Connection, *, session_id: str, **overrides: object) -> None:
    columns = {"session_id": session_id, **overrides}
    keys = list(columns.keys())
    placeholders = ", ".join("?" for _ in keys)
    conn.execute(
        f"INSERT INTO session_profiles ({', '.join(keys)}) VALUES ({placeholders})",
        tuple(columns.values()),
    )


def _insert_session_link(
    conn: sqlite3.Connection,
    *,
    child_session_id: str,
    dst_origin: str,
    dst_native_id: str,
    parent_session_id: str | None,
    branch_point_message_id: str | None = None,
    link_type: str = "subagent",
    status: str | None = None,
) -> None:
    """Canonical direction: the CHILD asserts the link (src_session_id), the
    PARENT is the resolved destination -- matching
    resolve_session_links_for_session, where `child_id =
    row["src_session_id"]` and the resolved session is written into
    `sessions.parent_session_id` keyed by that child. This is the reverse of
    the pre-y964 test fixtures, which is exactly the bug: those fixtures
    matched the (wrong) shipped view, not real ingestion."""
    conn.execute(
        """
        INSERT INTO session_links (
            src_session_id, dst_origin, dst_native_id, link_type,
            resolved_dst_session_id, branch_point_message_id, status, observed_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            child_session_id,
            dst_origin,
            dst_native_id,
            link_type,
            parent_session_id,
            branch_point_message_id,
            status,
            1_767_225_600_000,
        ),
    )


class _AsyncSqliteAdapter:
    """Minimal aiosqlite-compatible wrapper around stdlib sqlite3, matching
    the pattern in tests/unit/insights/test_topology_cycle_rejection.py --
    lets a real production async query helper run against a plain sqlite3
    connection inside a synchronous test."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        self._conn = conn

    async def execute(self, sql: str, params: tuple[object, ...] = ()) -> _AsyncCursorAdapter:
        cursor = self._conn.execute(sql, params)
        return _AsyncCursorAdapter(cursor)

    def commit(self) -> None:
        self._conn.commit()


class _AsyncCursorAdapter:
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    async def fetchall(self) -> list[sqlite3.Row]:
        return list(self._cursor.fetchall())

    async def fetchone(self) -> sqlite3.Row | None:
        return self._cursor.fetchone()

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount


def test_delegation_resolves_with_canonical_child_to_parent_direction(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")

    parent_id = _insert_session(conn, native_id="parent")
    child_id = _insert_session(conn, native_id="child")
    dispatch_message_id = _insert_message(
        conn, session_id=parent_id, native_id="dispatch", position=0, model_name="claude-opus-4-8"
    )
    _insert_dispatch_action(
        conn,
        message_id=dispatch_message_id,
        session_id=parent_id,
        position=0,
        tool_id="task-1",
        tool_input='{"prompt": "audit the thing"}',
        result_text="3 gaps found",
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
        child_session_id=child_id,
        dst_origin="claude-code-session",
        dst_native_id="parent",
        parent_session_id=parent_id,
        branch_point_message_id=dispatch_message_id,
    )

    row = conn.execute("SELECT * FROM delegations WHERE parent_session_id = ?", (parent_id,)).fetchone()
    assert row is not None
    # The load-bearing direction assertion: parent_session_id must be the
    # session that DISPATCHED (has the Task action), not the one that was
    # dispatched to. Under the reversed pre-fix view, this row would not
    # exist at all under this query (parent_session_id would resolve to
    # child_id instead).
    assert row["parent_session_id"] == parent_id
    assert row["child_session_id"] == child_id
    assert row["mapping_state"] == "resolved"
    assert row["parent_session_dominant_model"] == "claude-opus-4-8"
    assert row["parent_session_dominant_model_family"] == "anthropic"
    assert row["parent_origin"] == "claude-code-session"
    assert row["parent_terminal_state"] == "clean_finish"
    assert row["child_session_dominant_model"] == "claude-haiku-4-5"
    assert row["child_session_dominant_model_family"] == "anthropic"
    assert row["child_cost_usd"] == pytest.approx(0.42)
    assert row["child_tokens"] == 1000 + 500 + 200 + 50
    assert row["child_wall_ms"] == 44_100
    assert row["child_terminal_state"] == "clean_finish"
    assert row["instruction_payload"] == '{"prompt": "audit the thing"}'
    assert row["dispatch_turn_model"] == "claude-opus-4-8"
    assert row["artifact_text"] == "3 gaps found"
    assert row["result_is_error"] == 0
    assert row["result_exit_code"] == 0
    assert row["result_status"] == "ok"


def test_delegation_result_status_error_when_dispatch_action_reports_error(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="parent")
    child_id = _insert_session(conn, native_id="child")
    dispatch_message_id = _insert_message(conn, session_id=parent_id, native_id="dispatch", position=0)
    _insert_dispatch_action(
        conn,
        message_id=dispatch_message_id,
        session_id=parent_id,
        position=0,
        tool_id="task-1",
        result_text="boom",
        result_is_error=1,
        result_exit_code=1,
    )
    _insert_session_link(
        conn,
        child_session_id=child_id,
        dst_origin="claude-code-session",
        dst_native_id="parent",
        parent_session_id=parent_id,
        branch_point_message_id=dispatch_message_id,
    )

    row = conn.execute("SELECT result_status FROM delegations WHERE parent_session_id = ?", (parent_id,)).fetchone()
    assert row["result_status"] == "error"


def test_delegation_unresolved_when_dispatch_has_no_child_link(tmp_path: Path) -> None:
    """A dispatch error before child creation: one attempt, mapping_state
    unresolved, never zero rows and never a fabricated child."""
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="parent")
    dispatch_message_id = _insert_message(conn, session_id=parent_id, native_id="dispatch", position=0)
    _insert_dispatch_action(
        conn,
        message_id=dispatch_message_id,
        session_id=parent_id,
        position=0,
        tool_id="task-1",
        result_text=None,
    )
    # No session_links row at all -- the dispatch never produced a resolvable child.

    rows = conn.execute("SELECT * FROM delegations WHERE parent_session_id = ?", (parent_id,)).fetchall()
    assert len(rows) == 1
    assert rows[0]["mapping_state"] == "unresolved"
    assert rows[0]["child_session_id"] is None
    assert rows[0]["result_status"] == "unknown"


def test_delegation_fresh_spawned_child_with_null_branch_point_resolves(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="parent")
    child_id = _insert_session(conn, native_id="child")
    dispatch_message_id = _insert_message(conn, session_id=parent_id, native_id="dispatch", position=0)
    _insert_dispatch_action(conn, message_id=dispatch_message_id, session_id=parent_id, position=0, tool_id="task-1")
    _insert_session_link(
        conn,
        child_session_id=child_id,
        dst_origin="claude-code-session",
        dst_native_id="parent",
        parent_session_id=parent_id,
        branch_point_message_id=None,  # spawned-fresh: no inherited prefix
    )

    row = conn.execute("SELECT * FROM delegations WHERE parent_session_id = ?", (parent_id,)).fetchone()
    assert row is not None
    assert row["mapping_state"] == "resolved"
    assert row["child_session_id"] == child_id
    assert row["branch_point_message_id"] is None


def test_delegation_two_dispatches_in_one_message_no_fanout(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="parent")
    child_a = _insert_session(conn, native_id="child-a")
    child_b = _insert_session(conn, native_id="child-b")
    dispatch_message_id = _insert_message(conn, session_id=parent_id, native_id="dispatch", position=0)
    _insert_dispatch_action(
        conn, message_id=dispatch_message_id, session_id=parent_id, position=0, tool_id="task-1", result_text="a done"
    )
    _insert_dispatch_action(
        conn, message_id=dispatch_message_id, session_id=parent_id, position=2, tool_id="task-2", result_text="b done"
    )
    _insert_session_link(
        conn,
        child_session_id=child_a,
        dst_origin="claude-code-session",
        dst_native_id="parent",
        parent_session_id=parent_id,
    )
    _insert_session_link(
        conn,
        child_session_id=child_b,
        dst_origin="claude-code-session",
        dst_native_id="parent",
        parent_session_id=parent_id,
    )

    rows = conn.execute(
        "SELECT * FROM delegations WHERE parent_session_id = ? ORDER BY instruction_tool_use_block_id", (parent_id,)
    ).fetchall()
    assert len(rows) == 2
    assert {row["mapping_state"] for row in rows} == {"resolved"}
    assert {row["child_session_id"] for row in rows} == {child_a, child_b}
    assert {row["artifact_text"] for row in rows} == {"a done", "b done"}


def test_delegation_ambiguous_when_dispatch_and_child_counts_mismatch(tmp_path: Path) -> None:
    """Two dispatch actions but only one resolved child: rank-pairing would
    have to guess which dispatch produced the resolved child, so both rows
    surface as ambiguous with a real instruction but no fabricated winner."""
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="parent")
    child_id = _insert_session(conn, native_id="child")
    dispatch_message_id = _insert_message(conn, session_id=parent_id, native_id="dispatch", position=0)
    _insert_dispatch_action(conn, message_id=dispatch_message_id, session_id=parent_id, position=0, tool_id="task-1")
    _insert_dispatch_action(conn, message_id=dispatch_message_id, session_id=parent_id, position=2, tool_id="task-2")
    _insert_session_link(
        conn,
        child_session_id=child_id,
        dst_origin="claude-code-session",
        dst_native_id="parent",
        parent_session_id=parent_id,
    )

    rows = conn.execute("SELECT * FROM delegations WHERE parent_session_id = ?", (parent_id,)).fetchall()
    assert len(rows) == 2
    for row in rows:
        assert row["mapping_state"] == "ambiguous"
        assert row["child_session_id"] is None
        assert row["instruction_tool_use_block_id"] is not None


def test_delegation_edge_only_when_no_dispatch_action(tmp_path: Path) -> None:
    """Codex async subagents/sidechains with no parent Task action: counted,
    never given a fabricated instruction."""
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="parent", origin="codex-session")
    child_id = _insert_session(conn, native_id="child", origin="codex-session")

    _insert_session_link(
        conn,
        child_session_id=child_id,
        dst_origin="codex-session",
        dst_native_id="parent",
        parent_session_id=parent_id,
    )

    row = conn.execute("SELECT * FROM delegations WHERE parent_session_id = ?", (parent_id,)).fetchone()
    assert row is not None
    assert row["mapping_state"] == "edge_only"
    assert row["child_session_id"] == child_id
    assert row["result_status"] == "unknown"
    assert row["instruction_tool_use_block_id"] is None
    assert row["instruction_payload"] is None


def test_delegation_quarantined_link_surfaces_as_quarantined_state(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="parent")
    child_id = _insert_session(conn, native_id="child")

    _insert_session_link(
        conn,
        child_session_id=child_id,
        dst_origin="claude-code-session",
        dst_native_id="parent",
        parent_session_id=parent_id,
        status="quarantined",
    )

    rows = conn.execute("SELECT * FROM delegations WHERE parent_session_id = ?", (parent_id,)).fetchall()
    assert len(rows) == 1
    assert rows[0]["mapping_state"] == "quarantined"
    assert rows[0]["instruction_payload"] is None


def test_delegation_excludes_non_subagent_link_types(tmp_path: Path) -> None:
    """A prefix-sharing continuation/fork link is not a delegation."""
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="parent")
    child_id = _insert_session(conn, native_id="child")

    _insert_session_link(
        conn,
        child_session_id=child_id,
        dst_origin="claude-code-session",
        dst_native_id="parent",
        parent_session_id=parent_id,
        link_type="continuation",
    )

    rows = conn.execute("SELECT * FROM delegations WHERE parent_session_id = ?", (parent_id,)).fetchall()
    assert rows == []


def test_delegation_separates_dispatch_requested_and_child_observed_model_identity(tmp_path: Path) -> None:
    """polylogue-4c27: dispatch-turn model, requested model, and
    child-observed model must be independently readable and allowed to
    disagree -- none may silently overwrite another."""
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="parent")
    child_id = _insert_session(conn, native_id="child")
    dispatch_message_id = _insert_message(
        conn, session_id=parent_id, native_id="dispatch", position=0, model_name="claude-sonnet-4-6"
    )
    _insert_dispatch_action(
        conn,
        message_id=dispatch_message_id,
        session_id=parent_id,
        position=0,
        tool_id="task-1",
        tool_input='{"prompt": "route to a cheaper model", "model": "claude-haiku-4-5"}',
    )
    # The parent session as a whole is dominated by a different model than
    # the one that actually authored this dispatch turn.
    _insert_session_profile(
        conn, session_id=parent_id, primary_model_name="claude-opus-4-8", primary_model_family="anthropic"
    )
    _insert_session_profile(
        conn, session_id=child_id, primary_model_name="claude-fable-5", primary_model_family="anthropic"
    )
    _insert_session_link(
        conn,
        child_session_id=child_id,
        dst_origin="claude-code-session",
        dst_native_id="parent",
        parent_session_id=parent_id,
    )

    row = conn.execute("SELECT * FROM delegations WHERE parent_session_id = ?", (parent_id,)).fetchone()
    assert row is not None
    # Four genuinely distinct identities, none silently collapsed together:
    assert row["dispatch_turn_model"] == "claude-sonnet-4-6"
    assert row["requested_model"] == "claude-haiku-4-5"
    assert row["child_session_dominant_model"] == "claude-fable-5"
    assert row["parent_session_dominant_model"] == "claude-opus-4-8"
    values = {
        row["dispatch_turn_model"],
        row["requested_model"],
        row["child_session_dominant_model"],
        row["parent_session_dominant_model"],
    }
    assert len(values) == 4


def test_delegation_requested_model_unknown_when_not_recorded(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="parent")
    dispatch_message_id = _insert_message(conn, session_id=parent_id, native_id="dispatch", position=0)
    _insert_dispatch_action(
        conn,
        message_id=dispatch_message_id,
        session_id=parent_id,
        position=0,
        tool_id="task-1",
        tool_input='{"prompt": "no explicit route"}',
    )
    row = conn.execute("SELECT requested_model FROM delegations WHERE parent_session_id = ?", (parent_id,)).fetchone()
    assert row["requested_model"] is None


def test_delegation_direction_matches_real_link_resolver(tmp_path: Path) -> None:
    """Real-route regression: drive the ACTUAL production write path
    (upsert_session_links / resolve_session_links_for_session -- the same
    functions the daemon calls after parsing a session) instead of a
    hand-built row shape, then confirm the view reads parent/child in the
    correct direction against whatever the real resolver produced. This is
    the test that would fail outright against the pre-y964 reversed view."""
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="parent", origin="codex-session")
    child_id = _insert_session(conn, native_id="child", origin="codex-session")
    dispatch_message_id = _insert_message(conn, session_id=parent_id, native_id="dispatch", position=0)
    _insert_dispatch_action(conn, message_id=dispatch_message_id, session_id=parent_id, position=0, tool_id="task-1")

    adapter = _AsyncSqliteAdapter(conn)
    # The CHILD is the one that asserts the (initially unresolved) link to
    # its parent -- mirroring what a real subagent-session parser does.
    edge = TopologyEdgeRecord(
        src_session_id=SessionId(child_id),
        dst_origin=Origin.CODEX_SESSION,
        dst_native_id="parent",
        link_type=TopologyEdgeType.SUBAGENT,
        status=TopologyEdgeStatus.UNRESOLVED,
    )
    asyncio.run(upsert_session_links(adapter, [edge]))  # type: ignore[arg-type]
    resolved_count = asyncio.run(
        resolve_session_links_for_session(
            adapter,  # type: ignore[arg-type]
            session_id=parent_id,
            origin="codex-session",
            native_id="parent",
            resolved_at="2026-07-10T00:00:00+00:00",
        )
    )
    conn.commit()
    assert resolved_count == 1

    # The resolver must have written the PARENT into sessions.parent_session_id
    # keyed by the CHILD -- confirming our fixture direction matches reality.
    sessions_row = conn.execute("SELECT parent_session_id FROM sessions WHERE session_id = ?", (child_id,)).fetchone()
    assert sessions_row["parent_session_id"] == parent_id

    row = conn.execute("SELECT * FROM delegations WHERE parent_session_id = ?", (parent_id,)).fetchone()
    assert row is not None
    assert row["parent_session_id"] == parent_id
    assert row["child_session_id"] == child_id
    assert row["mapping_state"] == "resolved"
