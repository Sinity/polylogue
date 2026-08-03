"""polylogue-y964 / polylogue-4c27: the `delegations` view composes a
parent-dispatched subagent attempt from the PARENT's own dispatch actions
(`actions` rows, semantic_type='subagent'), corroborated against resolved
children via canonical `session_links` (child in `src_session_id`, parent in
`resolved_dst_session_id` -- see ``_resolve_outbound_session_links``,
``storage/sqlite/archive_tiers/write.py``). The prior shipped view aliased
these backwards; these fixtures use the canonical direction throughout and
would fail against that reversed view. Model identity is separated into
dispatch-turn / requested / child-observed / session-dominant-fallback
columns rather than one "orchestrator model"."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.query.unit_results import query_unit_envelope, query_unit_request
from polylogue.core.enums import BranchType, Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.surfaces.payloads import DelegationCardPayload, QueryUnitAggregateRowPayload

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
    branch_type: str | None = None,
    parent_session_id: str | None = None,
) -> str:
    conn.execute(
        """
        INSERT INTO sessions (
            native_id, origin, title, content_hash, created_at_ms, updated_at_ms,
            branch_type, parent_session_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            native_id,
            origin,
            f"session {native_id}",
            _HASH,
            created_at_ms,
            created_at_ms + 1000,
            branch_type,
            parent_session_id,
        ),
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
    ``_resolve_outbound_session_links`` (``storage/sqlite/archive_tiers/write.py``),
    where `child_id = row["src_session_id"]` and the resolved session is
    written into `sessions.parent_session_id` keyed by that child. This is
    the reverse of the pre-y964 test fixtures, which is exactly the bug: those fixtures
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


def _insert_child_first_message(conn: sqlite3.Connection, *, session_id: str, text: str) -> None:
    """Insert the child's own first user turn -- for a resolved subagent this
    IS (byte for byte) the dispatching Task's own prompt/message field; see
    polylogue-1vpm.7. Used to drive the view's content-identity join."""
    message_id = _insert_message(conn, session_id=session_id, native_id="first-turn", position=0)
    conn.execute(
        "UPDATE messages SET role = 'user', message_type = 'message' WHERE message_id = ?",
        (message_id,),
    )
    conn.execute(
        "INSERT INTO blocks (message_id, session_id, position, block_type, text) VALUES (?, ?, 0, 'text', ?)",
        (message_id, session_id, text),
    )


def test_delegation_two_dispatches_in_one_message_no_fanout(tmp_path: Path) -> None:
    """Two parallel dispatches in one cohort disambiguate by content
    identity, not by transcript order -- each dispatch's own prompt matches
    exactly one child's first turn."""
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="parent")
    child_a = _insert_session(conn, native_id="child-a")
    child_b = _insert_session(conn, native_id="child-b")
    dispatch_message_id = _insert_message(conn, session_id=parent_id, native_id="dispatch", position=0)
    _insert_dispatch_action(
        conn,
        message_id=dispatch_message_id,
        session_id=parent_id,
        position=0,
        tool_id="task-1",
        tool_input=json.dumps({"prompt": "audit module a"}),
        result_text="a done",
    )
    _insert_dispatch_action(
        conn,
        message_id=dispatch_message_id,
        session_id=parent_id,
        position=2,
        tool_id="task-2",
        tool_input=json.dumps({"prompt": "audit module b"}),
        result_text="b done",
    )
    _insert_child_first_message(conn, session_id=child_a, text="audit module a")
    _insert_child_first_message(conn, session_id=child_b, text="audit module b")
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
    # The pairing must respect CONTENT, not just presence of a match: the
    # dispatch whose own prompt is "audit module a" must resolve to child_a
    # specifically, not whichever child happens to sort first.
    by_prompt = {row["instruction_payload"]: row["child_session_id"] for row in rows}
    assert by_prompt[json.dumps({"prompt": "audit module a"})] == child_a
    assert by_prompt[json.dumps({"prompt": "audit module b"})] == child_b


def test_delegation_dispatch_without_matching_content_stays_unresolved(tmp_path: Path) -> None:
    """polylogue-1vpm.7 AC2/AC3: a parent with N dispatches and M<N captured
    children (here N=2, M=1) yields exactly M resolved and N-M unresolved --
    never N 'ambiguous'. Rank-pairing would have guessed a winner for both
    dispatches; content identity resolves only the one whose own prompt
    matches the captured child, and leaves the other honestly unresolved."""
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
        tool_input=json.dumps({"prompt": "captured dispatch"}),
    )
    _insert_dispatch_action(
        conn,
        message_id=dispatch_message_id,
        session_id=parent_id,
        position=2,
        tool_id="task-2",
        tool_input=json.dumps({"prompt": "never-captured dispatch"}),
    )
    _insert_child_first_message(conn, session_id=child_id, text="captured dispatch")
    _insert_session_link(
        conn,
        child_session_id=child_id,
        dst_origin="claude-code-session",
        dst_native_id="parent",
        parent_session_id=parent_id,
    )
    rows = conn.execute(
        "SELECT * FROM delegations WHERE parent_session_id = ? ORDER BY instruction_tool_use_block_id", (parent_id,)
    ).fetchall()
    assert len(rows) == 2
    states = {row["instruction_payload"]: row["mapping_state"] for row in rows}
    assert set(states.values()) == {"resolved", "unresolved"}, states
    resolved_row = next(row for row in rows if row["mapping_state"] == "resolved")
    assert resolved_row["child_session_id"] == child_id
    assert resolved_row["instruction_payload"] == json.dumps({"prompt": "captured dispatch"})
    unresolved_row = next(row for row in rows if row["mapping_state"] == "unresolved")
    assert unresolved_row["child_session_id"] is None
    assert unresolved_row["instruction_payload"] == json.dumps({"prompt": "never-captured dispatch"})


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
    (``write_parsed_session_to_archive`` -> ``_resolve_outbound_session_links``,
    ``storage/sqlite/archive_tiers/write.py`` -- the sole production writer,
    the same path the daemon calls after parsing a session) instead of a
    hand-built row shape, then confirm the view reads parent/child in the
    correct direction against whatever the real resolver produced. This is
    the test that would fail outright against the pre-y964 reversed view.

    polylogue-4ts.10: previously drove ``queries/session_links.py``'s
    ``upsert_session_links``/``resolve_session_links_for_session`` under the
    same "ACTUAL production write path" claim -- a 2026-08-03 structural
    audit found that engine has zero production callers and was deleted;
    this test now drives the real writer directly."""
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="parent", origin="codex-session")
    dispatch_message_id = _insert_message(conn, session_id=parent_id, native_id="dispatch", position=0)
    _insert_dispatch_action(conn, message_id=dispatch_message_id, session_id=parent_id, position=0, tool_id="task-1")
    conn.commit()

    # The CHILD is the one that asserts the (initially unresolved) link to
    # its parent -- mirroring what a real subagent-session parser does.
    child_session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        parent_session_provider_id="parent",
        branch_type=BranchType.SUBAGENT,
        messages=[ParsedMessage(provider_message_id="c0", role=Role.USER, text="go", position=0)],
    )
    child_id = write_parsed_session_to_archive(conn, child_session, content_hash=session_content_hash(child_session))
    conn.commit()

    # The resolver must have written the PARENT into sessions.parent_session_id
    # keyed by the CHILD -- confirming our fixture direction matches reality.
    sessions_row = conn.execute("SELECT parent_session_id FROM sessions WHERE session_id = ?", (child_id,)).fetchone()
    assert sessions_row["parent_session_id"] == parent_id

    link_row = conn.execute(
        "SELECT status, resolved_dst_session_id FROM session_links WHERE src_session_id = ?", (child_id,)
    ).fetchone()
    assert link_row is not None
    assert link_row["status"] is None
    assert link_row["resolved_dst_session_id"] == parent_id

    row = conn.execute("SELECT * FROM delegations WHERE parent_session_id = ?", (parent_id,)).fetchone()
    assert row is not None
    assert row["parent_session_id"] == parent_id
    assert row["child_session_id"] == child_id
    assert row["mapping_state"] == "resolved"


def test_delegation_query_unit_and_card_use_real_attempt_relation(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="parent")
    child_id = _insert_session(conn, native_id="child", branch_type="subagent", parent_session_id=parent_id)
    context_message_ids: list[str] = []
    for position in range(4):
        message_id = _insert_message(
            conn,
            session_id=parent_id,
            native_id=f"context-{position}",
            position=position,
        )
        context_message_ids.append(message_id)
        conn.execute(
            "INSERT INTO blocks (message_id, session_id, position, block_type, text) VALUES (?, ?, 0, 'text', ?)",
            (message_id, parent_id, f"bounded context {position}"),
        )
    dispatch_message_id = _insert_message(
        conn,
        session_id=parent_id,
        native_id="dispatch",
        position=4,
        model_name="claude-opus-4-8",
    )
    instruction = "review the complete subsystem " + "carefully " * 40
    _insert_dispatch_action(
        conn,
        message_id=dispatch_message_id,
        session_id=parent_id,
        position=0,
        tool_id="task-1",
        tool_input=json.dumps({"prompt": instruction, "model": "haiku"}),
        result_text="three bounded findings",
    )
    followup_message_ids: list[str] = []
    for position in range(5, 9):
        message_id = _insert_message(
            conn,
            session_id=parent_id,
            native_id=f"followup-{position}",
            position=position,
        )
        followup_message_ids.append(message_id)
        conn.execute(
            "INSERT INTO blocks (message_id, session_id, position, block_type, text) VALUES (?, ?, 0, 'text', ?)",
            (message_id, parent_id, f"parent followup {position}"),
        )
    child_message_id = _insert_message(conn, session_id=child_id, native_id="child-result", position=0)
    conn.execute(
        "INSERT INTO blocks (message_id, session_id, position, block_type, text) VALUES (?, ?, 0, 'text', ?)",
        (child_message_id, child_id, "actual child findings"),
    )
    _insert_session_link(
        conn,
        child_session_id=child_id,
        dst_origin="claude-code-session",
        dst_native_id="parent",
        parent_session_id=parent_id,
    )
    conn.commit()
    conn.close()

    initialize_archive_database(tmp_path / "user.db", ArchiveTier.USER)
    with ArchiveStore.open_existing(tmp_path) as archive:
        envelope = query_unit_envelope(
            archive,
            query_unit_request(
                expression="delegations where mapping_state:resolved AND instruction:subsystem",
                limit=10,
            ),
        )
        assert envelope.unit == "delegation"
        [item] = envelope.items
        payload = item.model_dump(mode="json")
        assert payload["mapping_state"] == "resolved"
        assert payload["parent_session_id"] == parent_id
        assert payload["child_session_id"] == child_id
        assert payload["instruction_preview"] == instruction[:240]
        assert payload["instruction_sha256"] == hashlib.sha256(instruction.encode()).hexdigest()
        assert payload["instruction_truncated"] is True
        assert "instruction_payload" not in payload
        assert "child_cost_usd" not in payload

        counts = query_unit_envelope(
            archive,
            query_unit_request(
                expression="delegations where instruction:subsystem | group by mapping_state | count",
                limit=10,
            ),
        )
        aggregate_rows = [row for row in counts.items if isinstance(row, QueryUnitAggregateRowPayload)]
        assert len(aggregate_rows) == len(counts.items)
        assert [(row.group_key, row.count) for row in aggregate_rows] == [("resolved", 1)]

        card = archive.get_delegation_card(instruction_tool_use_block_id=f"{dispatch_message_id}:0")
        assert card is not None
        assert card.instruction == instruction
        assert card.parent_session_title == "session parent"
        assert card.child_session_title == "session child"
        # run_relation_sql() (polylogue-dab) keys run_ref to the subagent's
        # own session_id, not the parent's -- see get_delegation_card's
        # native_session_id-matching comment in archive.py.
        assert card.run_ref == f"run:{child_id}"
        assert card.run_title == f"session {child_id.split(':', 1)[1]}"
        assert card.dispatch_result == "three bounded findings"
        assert card.dispatch_result_truncated is False
        assert card.child_excerpt == "actual child findings"
        assert card.child_excerpt_truncated is False
        assert [row.message_id for row in card.parent_context] == context_message_ids[1:]
        assert card.parent_context_truncated is True
        assert [row.message_id for row in card.parent_followup] == followup_message_ids[:3]
        assert card.parent_followup_truncated is True
        assert card.parent_followup[0].text == "parent followup 5"
        assert f"message:{context_message_ids[1]}" in card.evidence_refs
        assert f"message:{followup_message_ids[0]}" in card.evidence_refs
        assert f"message:{child_message_id}" in card.evidence_refs
        surface_card = DelegationCardPayload.from_card(card)
        assert surface_card.parent_context[0].message_ref == f"message:{context_message_ids[1]}"
        assert surface_card.parent_context_truncated is True
        assert surface_card.parent_followup[0].message_ref == f"message:{followup_message_ids[0]}"
        assert surface_card.parent_followup_truncated is True


def test_delegation_query_unit_keeps_edge_only_attempts_honest(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="edge-parent")
    child_id = _insert_session(conn, native_id="edge-child")
    _insert_session_link(
        conn,
        child_session_id=child_id,
        dst_origin="claude-code-session",
        dst_native_id="edge-parent",
        parent_session_id=parent_id,
    )
    conn.commit()
    conn.close()

    initialize_archive_database(tmp_path / "user.db", ArchiveTier.USER)
    with ArchiveStore.open_existing(tmp_path) as archive:
        envelope = query_unit_envelope(
            archive,
            query_unit_request(expression="delegations where mapping_state:edge_only", limit=10),
        )
        [item] = envelope.items
        payload = item.model_dump(mode="json")
        assert payload["parent_session_id"] == parent_id
        assert payload["child_session_id"] == child_id
        assert payload["evidence_basis"] == "edge"
        assert payload["instruction_preview"] is None
        assert payload["instruction_sha256"] is None
        assert payload["instruction_tool_use_block_id"] is None
        assert payload["result_status"] == "unknown"


def test_delegation_instruction_filter_matches_preview_extraction(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="instruction-parent")
    payloads = (
        ("empty", "{}"),
        ("fallback", json.dumps({"prompt": "", "description": "review fallback"})),
        ("numeric", json.dumps({"prompt": 7, "description": "review numeric fallback"})),
    )
    for position, (native_id, payload) in enumerate(payloads):
        message_id = _insert_message(
            conn,
            session_id=parent_id,
            native_id=native_id,
            position=position,
        )
        _insert_dispatch_action(
            conn,
            message_id=message_id,
            session_id=parent_id,
            position=0,
            tool_id=f"task-{native_id}",
            tool_input=payload,
            result_text=None,
        )
    conn.commit()
    conn.close()

    initialize_archive_database(tmp_path / "user.db", ArchiveTier.USER)
    with ArchiveStore.open_existing(tmp_path) as archive:
        envelope = query_unit_envelope(
            archive,
            query_unit_request(expression="delegations where instruction:review", limit=10),
        )
        previews = {item.model_dump(mode="json")["instruction_preview"] for item in envelope.items}
        assert previews == {"review fallback", "review numeric fallback"}

        empty = query_unit_envelope(
            archive,
            query_unit_request(expression="delegations where parent:instruction-parent", limit=10),
        )
        empty_payload = next(
            item.model_dump(mode="json")
            for item in empty.items
            if item.model_dump(mode="json").get("instruction_tool_use_block_id") == f"{parent_id}:empty:0"
        )
        assert empty_payload["instruction_preview"] is None


# ---------------------------------------------------------------------------
# polylogue-qsb4: arbitrary-depth ancestry/subtree recursive queries.
# ---------------------------------------------------------------------------


def _dispatch_chain_level(
    conn: sqlite3.Connection,
    *,
    dispatcher_id: str,
    dispatcher_native_id: str,
    child_native_id: str,
    tool_id: str,
) -> str:
    """Create one trivial-cohort delegation edge: ``dispatcher_id`` has
    exactly one dispatch action and exactly one resolved child, so the view
    pairs them as ``mapping_state='resolved'`` without needing content-
    identity disambiguation (the trivial-cohort case, see
    ``delegation_facts_source`` in index.py). Returns the new child's
    session id."""

    child_id = _insert_session(conn, native_id=child_native_id)
    message_id = _insert_message(conn, session_id=dispatcher_id, native_id=f"dispatch-{tool_id}", position=0)
    _insert_dispatch_action(
        conn,
        message_id=message_id,
        session_id=dispatcher_id,
        position=0,
        tool_id=tool_id,
        tool_input="{}",
        result_text="done",
    )
    _insert_session_link(
        conn,
        child_session_id=child_id,
        dst_origin="claude-code-session",
        dst_native_id=dispatcher_native_id,
        parent_session_id=dispatcher_id,
    )
    return child_id


def test_delegation_ancestry_returns_root_to_node_chain_depth_annotated(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    root_id = _insert_session(conn, native_id="root")
    mid_id = _dispatch_chain_level(
        conn, dispatcher_id=root_id, dispatcher_native_id="root", child_native_id="mid", tool_id="task-root-mid"
    )
    leaf_id = _dispatch_chain_level(
        conn, dispatcher_id=mid_id, dispatcher_native_id="mid", child_native_id="leaf", tool_id="task-mid-leaf"
    )
    conn.commit()
    conn.close()

    initialize_archive_database(tmp_path / "user.db", ArchiveTier.USER)
    with ArchiveStore.open_existing(tmp_path) as archive:
        ancestry = archive.get_delegation_ancestry(leaf_id)

    # Root-first (depth descending), leaf itself last at depth 0 -- no N+1,
    # one recursive-CTE call produced the whole chain.
    assert [node.session_id for node in ancestry] == [root_id, mid_id, leaf_id]
    assert [node.depth for node in ancestry] == [2, 1, 0]
    assert ancestry[0].child_session_id == mid_id
    assert ancestry[0].mapping_state == "resolved"
    assert ancestry[1].child_session_id == leaf_id
    assert ancestry[1].mapping_state == "resolved"
    assert ancestry[2].child_session_id is None
    assert ancestry[2].mapping_state is None

    # A session nobody ever dispatched returns just itself.
    with ArchiveStore.open_existing(tmp_path) as archive:
        root_ancestry = archive.get_delegation_ancestry(root_id)
    assert [node.session_id for node in root_ancestry] == [root_id]
    assert root_ancestry[0].depth == 0


def test_delegation_subtree_returns_all_descendants_depth_annotated(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    root_id = _insert_session(conn, native_id="root")
    child_a = _dispatch_chain_level(
        conn, dispatcher_id=root_id, dispatcher_native_id="root", child_native_id="child-a", tool_id="task-a"
    )
    grandchild = _dispatch_chain_level(
        conn, dispatcher_id=child_a, dispatcher_native_id="child-a", child_native_id="grandchild", tool_id="task-b"
    )
    conn.commit()
    conn.close()

    initialize_archive_database(tmp_path / "user.db", ArchiveTier.USER)
    with ArchiveStore.open_existing(tmp_path) as archive:
        subtree = archive.get_delegation_subtree(root_id)

    by_session = {node.session_id: node for node in subtree}
    assert set(by_session) == {root_id, child_a, grandchild}
    assert by_session[root_id].depth == 0
    assert by_session[root_id].parent_session_id is None
    assert by_session[child_a].depth == 1
    assert by_session[child_a].parent_session_id == root_id
    assert by_session[child_a].mapping_state == "resolved"
    assert by_session[grandchild].depth == 2
    assert by_session[grandchild].parent_session_id == child_a

    # A leaf that dispatched nothing returns just itself.
    with ArchiveStore.open_existing(tmp_path) as archive:
        leaf_subtree = archive.get_delegation_subtree(grandchild)
    assert [node.session_id for node in leaf_subtree] == [grandchild]


def test_delegation_subtree_excludes_quarantined_edges(tmp_path: Path) -> None:
    """Quarantined edges (session_links' TopologyEdgeStatus cycle-break
    precedent) are structural cycle-breaks and must never be traversed --
    the recursive CTE reuses that vocabulary rather than re-deriving it."""

    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="quarantine-parent")
    child_id = _insert_session(conn, native_id="quarantine-child")
    _insert_session_link(
        conn,
        child_session_id=child_id,
        dst_origin="claude-code-session",
        dst_native_id="quarantine-parent",
        parent_session_id=parent_id,
        status="quarantined",
    )
    conn.commit()
    conn.close()

    initialize_archive_database(tmp_path / "user.db", ArchiveTier.USER)
    with ArchiveStore.open_existing(tmp_path) as archive:
        subtree = archive.get_delegation_subtree(parent_id)
        ancestry = archive.get_delegation_ancestry(child_id)

    # Only the queried node itself -- the quarantined edge is never followed.
    assert [node.session_id for node in subtree] == [parent_id]
    assert [node.session_id for node in ancestry] == [child_id]


def test_delegation_subtree_visited_path_guard_stops_a_two_node_cycle(tmp_path: Path) -> None:
    """Two independent trivial-cohort (non-quarantined, `mapping_state=
    'resolved'`) edges can compose into a cycle that session_links' own
    quarantine pass never inspects (quarantine is asserted per-edge over
    session_links alone, not over the composed `delegations` chain). The
    recursive CTE's own defensive visited-path guard must still terminate
    -- this is the exact scenario polylogue-qsb4 AC3 requires an explicit
    answer for."""

    conn = _connect(tmp_path / "index.db")
    a_id = _insert_session(conn, native_id="cycle-a")
    b_id = _insert_session(conn, native_id="cycle-b")

    message_a = _insert_message(conn, session_id=a_id, native_id="dispatch-a-b", position=0)
    _insert_dispatch_action(
        conn, message_id=message_a, session_id=a_id, position=0, tool_id="task-a-b", tool_input="{}"
    )
    _insert_session_link(
        conn, child_session_id=b_id, dst_origin="claude-code-session", dst_native_id="cycle-a", parent_session_id=a_id
    )

    message_b = _insert_message(conn, session_id=b_id, native_id="dispatch-b-a", position=0)
    _insert_dispatch_action(
        conn, message_id=message_b, session_id=b_id, position=0, tool_id="task-b-a", tool_input="{}"
    )
    _insert_session_link(
        conn, child_session_id=a_id, dst_origin="claude-code-session", dst_native_id="cycle-b", parent_session_id=b_id
    )
    conn.commit()
    conn.close()

    initialize_archive_database(tmp_path / "user.db", ArchiveTier.USER)
    with ArchiveStore.open_existing(tmp_path) as archive:
        # Both edges are genuinely `resolved` (neither quarantined) -- proof
        # the cycle survives topology's own quarantine pass and only the
        # recursive query's own guard stops it.
        subtree_from_a = archive.get_delegation_subtree(a_id)
        ancestry_from_a = archive.get_delegation_ancestry(a_id)

    # The guard must terminate (no RecursionError/timeout) and must not
    # revisit a session already on the current path.
    assert [node.session_id for node in subtree_from_a] == [a_id, b_id]
    assert [node.depth for node in subtree_from_a] == [0, 1]
    assert [node.session_id for node in ancestry_from_a] == [b_id, a_id]
    assert [node.depth for node in ancestry_from_a] == [1, 0]
