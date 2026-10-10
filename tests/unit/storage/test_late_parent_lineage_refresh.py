"""A late parent changes sessions other than the one being written.

When a child is archived before its parent, the parent's write resolves the
child's edge, deletes the child's inherited prefix and moves the child under
the parent's root. All of that happens under the session-write guard, which
keeps the block and link triggers from refreshing derived relations, so the
write itself must refresh every session it changed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_write
from polylogue.storage.sqlite.delegation_facts import rebuild_all_delegation_facts_sync
from tests.infra.index_writer import fixture_index_mutation_scope, write_fixture_index_session

_DISPATCH_TOOL_ID = "toolu_dispatch_survey"


def _connect(path: Path) -> sqlite3.Connection:
    conn = connect_measured(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _text(pid: str, role: Role, text: str, position: int) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=pid,
        role=role,
        text=text,
        position=position,
        variant_index=0,
        is_active_path=True,
        is_active_leaf=False,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _dispatch(pid: str, position: int) -> ParsedMessage:
    """A subagent dispatch: ``Task`` classifies as ``semantic_type='subagent'``."""
    return ParsedMessage(
        provider_message_id=pid,
        role=Role.ASSISTANT,
        text="dispatching a surveyor",
        position=position,
        variant_index=0,
        is_active_path=True,
        is_active_leaf=False,
        blocks=[
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_name="Task",
                tool_id=_DISPATCH_TOOL_ID,
                tool_input={"prompt": "survey the logs"},
            )
        ],
    )


def _assert_delegation_facts_match_a_rebuild(conn: sqlite3.Connection) -> None:
    """The per-write refresh left exactly what a from-scratch derivation produces."""

    def rows() -> list[tuple[object, ...]]:
        return [tuple(row) for row in conn.execute("SELECT * FROM delegation_facts ORDER BY delegation_id")]

    refreshed = rows()
    rebuild_all_delegation_facts_sync(conn)
    assert refreshed == rows()


def _session(native_id: str, messages: list[ParsedMessage], *, parent: str | None = None) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=native_id,
        title=native_id,
        parent_session_provider_id=parent,
        branch_type=BranchType.FORK if parent else None,
        messages=messages,
    )


@pytest.mark.parametrize("foreign_keys", [True, False], ids=["fk-on", "fk-suspended"])
def test_late_parent_refreshes_the_child_dispatch_cohort(tmp_path: Path, foreign_keys: bool) -> None:
    """The child's replayed dispatch leaves its cohort once the parent owns it.

    Red twin: drop the re-extracted child from the delegation refresh in
    ``write_parsed_session_to_archive``. The child's prefix blocks are deleted
    under the session-write guard, so its ``delegation_facts`` row keeps
    naming the deleted dispatch block. With foreign keys suspended (bulk
    ingest), dropping the child's ``refresh_action_pairs`` from
    ``_reextract_prefix_tail_db`` also leaves an ``action_pairs`` row on the
    deleted block, since no cascade removes it.
    """
    conn = _connect(tmp_path / "index.db")
    child_id = write_fixture_index_session(
        conn,
        _session(
            "child",
            [
                _text("c0", Role.USER, "hello", 0),
                _dispatch("c1", 1),
                _text("cx", Role.USER, "child diverges here", 2),
                _text("cy", Role.ASSISTANT, "child reply", 3),
            ],
            parent="parent",
        ),
    )
    before = conn.execute(
        "SELECT mapping_state, instruction_tool_use_block_id FROM delegation_facts WHERE parent_session_id = ?",
        (child_id,),
    ).fetchall()
    assert [row["mapping_state"] for row in before] == ["unresolved"], "the stored-whole child owns the dispatch"
    assert conn.execute("SELECT COUNT(*) FROM action_pairs WHERE session_id = ?", (child_id,)).fetchone()[0] == 1

    parent = _session(
        "parent",
        [
            _text("p0", Role.USER, "hello", 0),
            _dispatch("p1", 1),
            _text("p2", Role.USER, "parent continues alone", 2),
        ],
    )
    if foreign_keys:
        parent_id = write_fixture_index_session(conn, parent)
    else:
        # Bulk ingest suspends foreign keys and publishes a prepared carrier
        # inside its own explicitly owned Index transaction scope.
        conn.execute("PRAGMA foreign_keys = OFF")
        prepared = prepare_session_write(conn, parent, merge_append=False)
        try:
            with fixture_index_mutation_scope(conn):
                parent_id = write_fixture_index_session(conn, parent, prepared_write=prepared)
        finally:
            prepared.close()

    stored = conn.execute(
        "SELECT position FROM messages WHERE session_id = ? ORDER BY position", (child_id,)
    ).fetchall()
    assert [row[0] for row in stored] == [2, 3], "the late parent re-extracted the child to its tail"

    assert (
        conn.execute("SELECT COUNT(*) FROM delegation_facts WHERE parent_session_id = ?", (child_id,)).fetchone()[0]
        == 0
    )
    dangling_facts = conn.execute(
        """
        SELECT COUNT(*) FROM delegation_facts AS f
        WHERE (f.instruction_tool_use_block_id IS NOT NULL
               AND NOT EXISTS (SELECT 1 FROM blocks WHERE block_id = f.instruction_tool_use_block_id))
           OR (f.artifact_block_id IS NOT NULL
               AND NOT EXISTS (SELECT 1 FROM blocks WHERE block_id = f.artifact_block_id))
        """
    ).fetchone()[0]
    assert dangling_facts == 0
    dangling_pairs = conn.execute(
        """
        SELECT COUNT(*) FROM action_pairs AS ap
        WHERE NOT EXISTS (SELECT 1 FROM blocks WHERE block_id = ap.tool_use_block_id)
        """
    ).fetchone()[0]
    assert dangling_pairs == 0

    parent_facts = conn.execute(
        """
        SELECT f.mapping_state, b.session_id AS block_session_id
        FROM delegation_facts AS f JOIN blocks AS b ON b.block_id = f.instruction_tool_use_block_id
        WHERE f.parent_session_id = ?
        """,
        (parent_id,),
    ).fetchall()
    assert [(row["mapping_state"], row["block_session_id"]) for row in parent_facts] == [("unresolved", parent_id)]
    _assert_delegation_facts_match_a_rebuild(conn)
    if not foreign_keys:
        conn.rollback()
    conn.close()


def test_materialized_prefix_joins_the_child_dispatch_cohort(tmp_path: Path) -> None:
    """A child that copies its inherited prefix owns the dispatch in it.

    A parent rewrite that drops the child's branch point materializes the
    pre-write prefix into the child's own rows, inserting the dispatch block
    under the session-write guard. Red twin: leave the materialized session
    out of the delegation refresh; the child's cohort stays empty while its
    ``actions`` hold a subagent dispatch, and a rebuild disagrees.
    """
    conn = _connect(tmp_path / "index.db")
    parent_messages = [
        _text("p0", Role.USER, "hello", 0),
        _dispatch("p1", 1),
        _text("p2", Role.USER, "parent continues alone", 2),
    ]
    write_fixture_index_session(conn, _session("parent", parent_messages))
    child_id = write_fixture_index_session(
        conn,
        _session(
            "child",
            [
                _text("c0", Role.USER, "hello", 0),
                _dispatch("c1", 1),
                _text("cx", Role.USER, "child diverges here", 2),
                _text("cy", Role.ASSISTANT, "child reply", 3),
            ],
            parent="parent",
        ),
    )
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (child_id,)).fetchone()[0] == 2
    assert (
        conn.execute("SELECT COUNT(*) FROM delegation_facts WHERE parent_session_id = ?", (child_id,)).fetchone()[0]
        == 0
    )

    # The rewrite drops the child's branch point (the dispatch turn).
    write_fixture_index_session(
        conn, _session("parent", [parent_messages[0], _text("p3", Role.USER, "parent rewritten", 1)])
    )

    materialized = conn.execute(
        "SELECT COUNT(*) FROM blocks WHERE session_id = ? AND tool_id = ?", (child_id, _DISPATCH_TOOL_ID)
    ).fetchone()[0]
    assert materialized == 1, "the child now owns a copy of the dispatch turn"
    child_facts = conn.execute(
        """
        SELECT f.mapping_state, b.session_id AS block_session_id
        FROM delegation_facts AS f JOIN blocks AS b ON b.block_id = f.instruction_tool_use_block_id
        WHERE f.parent_session_id = ?
        """,
        (child_id,),
    ).fetchall()
    assert [(row["mapping_state"], row["block_session_id"]) for row in child_facts] == [("unresolved", child_id)]
    _assert_delegation_facts_match_a_rebuild(conn)
    conn.close()


def test_late_parent_moves_every_descendant_to_its_root(tmp_path: Path) -> None:
    """A grandchild written under a rootless child joins the late parent's thread.

    Red twin: drop ``_propagate_root_to_descendants`` from the projection. The
    grandchild keeps the child as its root, so ``threads`` reports the parent
    thread with two sessions and a second thread holding only the grandchild.
    """
    conn = _connect(tmp_path / "index.db")
    child_id = write_fixture_index_session(
        conn,
        _session(
            "child",
            [_text("b0", Role.USER, "child opens", 0), _text("b1", Role.ASSISTANT, "child answers", 1)],
            parent="parent",
        ),
    )
    grandchild_id = write_fixture_index_session(
        conn,
        _session(
            "grandchild",
            [_text("g0", Role.USER, "grandchild opens", 0), _text("g1", Role.ASSISTANT, "grandchild answers", 1)],
            parent="child",
        ),
    )
    roots = dict(conn.execute("SELECT session_id, root_session_id FROM sessions").fetchall())
    assert roots[grandchild_id] == child_id, "before the parent arrives the child is the thread root"

    parent_id = write_fixture_index_session(
        conn,
        _session(
            "parent",
            [_text("p0", Role.USER, "parent opens", 0), _text("p1", Role.ASSISTANT, "parent answers", 1)],
        ),
    )

    rows = {
        str(row["session_id"]): (row["parent_session_id"], row["root_session_id"])
        for row in conn.execute("SELECT session_id, parent_session_id, root_session_id FROM sessions")
    }
    assert rows == {
        parent_id: (None, parent_id),
        child_id: (parent_id, parent_id),
        grandchild_id: (child_id, parent_id),
    }
    threads = [tuple(row) for row in conn.execute("SELECT thread_id, session_count, depth FROM threads")]
    assert threads == [(parent_id, 3, 2)]
    conn.close()


def test_unchanged_lineage_projection_does_not_issue_session_updates(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers import write

    conn = _connect(tmp_path / "index.db")
    parent = write_fixture_index_session(conn, _session("no-op-parent", [_text("p", Role.USER, "parent", 0)]))
    child = write_fixture_index_session(
        conn, _session("no-op-child", [_text("c", Role.USER, "child", 0)], parent="no-op-parent")
    )
    grandchild = write_fixture_index_session(
        conn, _session("no-op-grandchild", [_text("g", Role.USER, "grandchild", 0)], parent="no-op-child")
    )
    before = [
        tuple(row)
        for row in conn.execute(
            "SELECT session_id,parent_session_id,root_session_id,branch_type,session_kind FROM sessions ORDER BY session_id"
        )
    ]
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        for session_id in (parent, child, grandchild):
            write._refresh_session_projection(conn, session_id, seen=set())
    finally:
        conn.set_trace_callback(None)
    assert not [sql for sql in statements if sql.lstrip().upper().startswith("UPDATE SESSIONS")]
    after = [
        tuple(row)
        for row in conn.execute(
            "SELECT session_id,parent_session_id,root_session_id,branch_type,session_kind FROM sessions ORDER BY session_id"
        )
    ]
    assert after == before
