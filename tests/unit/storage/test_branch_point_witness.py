"""A stale branch point is re-anchored only where its content witness agrees.

``_repair_stale_prefix_branch_points_db`` re-anchored a dangling prefix-sharing
edge onto whatever message carried the same native-id suffix in the parent's
composed transcript, without touching ``branch_point_content_address``. The
reader checks that witness, so it rejected the re-anchored edge and served the
child's bare tail. A re-anchor now requires the witness to agree; where it
does not, the child's inherited prefix is materialized into its own rows in
the same write (polylogue-gy2yu), so the child keeps its own content.

Anti-vacuity: drop the witness comparison and ``test_witness_mismatch_keeps_the_childs_own_content``
goes red -- the edge re-anchors onto ``codex-session:gp:n:m1`` and the child
reads ``other-m1``, content its own bytes never carried.
``test_witness_agreement_still_repairs`` pins the opposite direction, so
materializing on every relocation cannot pass.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Origin, Provider, WebConstructType
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.parsers.base import (
    ParsedContentBlock,
    ParsedFileEdit,
    ParsedMessage,
    ParsedSession,
    ParsedWebConstruct,
)
from polylogue.sources.prepared_message_sink import SqliteMessageSink, SqliteMessageStore
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    count_dangling_prefix_branch_points,
    read_archive_session_envelope,
)
from tests.infra.index_writer import write_fixture_index_session


def _connect(path: Path) -> sqlite3.Connection:
    conn = connect_measured(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _message(provider_message_id: str, text: str, position: int) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_message_id,
        role=Role.USER if position % 2 == 0 else Role.ASSISTANT,
        text=text,
        position=position,
        variant_index=0,
        is_active_path=True,
        is_active_leaf=False,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _session(session_id: str, pairs: list[tuple[str, str]], *, parent: str | None = None) -> ParsedSession:
    """A Codex session whose ``(native id, text)`` pairs are given separately.

    Keeping them separate is the point: the defect needs one message whose
    native id is reused across sessions while its content differs.
    """
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=session_id,
        title=session_id,
        parent_session_provider_id=parent,
        branch_type=BranchType.FORK if parent is not None else None,
        messages=[_message(pid, text, index) for index, (pid, text) in enumerate(pairs)],
    )


def _witness_session(
    session_id: str,
    *,
    extra_kind: str,
    extra_value: str,
    parent: str | None = None,
) -> ParsedSession:
    """Build the same branch transcript while changing one hashed block field."""
    if extra_kind == "metadata":
        block = ParsedContentBlock(type=BlockType.TEXT, text="m1", metadata={"revision": extra_value})
    elif extra_kind == "file_edit":
        block = ParsedContentBlock(
            type=BlockType.TOOL_RESULT,
            text="m1",
            tool_id="edit-1",
            is_error=False,
            file_edit=ParsedFileEdit(file_path="/tmp/witness.py", old_string=extra_value, new_string="after"),
        )
    elif extra_kind == "web_constructs":
        block = ParsedContentBlock(
            type=BlockType.TEXT,
            text="m1",
            web_constructs=[ParsedWebConstruct(construct_type=WebConstructType.CONTENT_REFERENCE, url=extra_value)],
        )
    else:
        raise AssertionError(extra_kind)
    first_message = _message("m0", "m0", 0)
    if extra_kind == "file_edit":
        first_message.blocks.append(
            ParsedContentBlock(type=BlockType.TOOL_USE, tool_id="edit-1", tool_name="Edit", tool_input={})
        )
    messages = [
        first_message,
        ParsedMessage(
            provider_message_id="m1",
            role=Role.ASSISTANT,
            parent_message_provider_id="m0" if extra_kind == "file_edit" else None,
            text="m1",
            position=1,
            variant_index=0,
            is_active_path=True,
            is_active_leaf=False,
            blocks=[block],
        ),
    ]
    if parent is not None:
        messages.append(_message("x", "x", 2))
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=session_id,
        title=session_id,
        parent_session_provider_id=parent,
        branch_type=BranchType.FORK if parent is not None else None,
        messages=messages,
    )


def _edge(conn: sqlite3.Connection, child_id: str) -> tuple[str, bytes]:
    row = conn.execute(
        "SELECT branch_point_message_id, branch_point_content_address FROM session_links WHERE src_session_id = ?",
        (child_id,),
    ).fetchone()
    return str(row[0]), bytes(row[1])


def _composed(conn: sqlite3.Connection, session_id: str) -> tuple[list[str | None], bool, str | None]:
    envelope = read_archive_session_envelope(conn, session_id)
    return (
        [message.blocks[0].text for message in envelope.messages],
        bool(envelope.lineage_complete),
        envelope.lineage_truncation_reason,
    )


class TestBranchPointWitness:
    def test_exact_hashed_block_fields_invalidate_the_branch_witness(self, tmp_path: Path) -> None:
        """Metadata, edits, and web constructs all participate in the witness."""
        for extra_kind in ("metadata", "file_edit", "web_constructs"):
            root = tmp_path / extra_kind
            root.mkdir()
            db = root / "index.db"
            conn = _connect(db)
            try:
                parent = _witness_session("parent", extra_kind=extra_kind, extra_value="before")
                child = _witness_session("child", extra_kind=extra_kind, extra_value="before", parent="parent")
                write_fixture_index_session(conn, parent)
                child_id = write_fixture_index_session(conn, child)
                conn.commit()
                assert _composed(conn, child_id) == (["m0", "m1", "x"], True, None)

                write_fixture_index_session(
                    conn,
                    _witness_session("parent", extra_kind=extra_kind, extra_value="after"),
                )
                conn.commit()

                edge = conn.execute(
                    "SELECT inheritance FROM session_links WHERE src_session_id = ?", (child_id,)
                ).fetchone()
                assert edge is not None and edge[0] == "spawned-fresh"
                composed = read_archive_session_envelope(conn, child_id)
                branch_message = composed.messages[1]
                block = branch_message.blocks[0]
                row = conn.execute(
                    "SELECT semantic_extra_json FROM blocks WHERE block_id = ?", (block.block_id,)
                ).fetchone()
                assert row is not None and row[0] is not None
                extras = json.loads(row[0])
                if extra_kind == "metadata":
                    assert extras["metadata"] == {"revision": "before"}
                elif extra_kind == "file_edit":
                    assert extras["file_edit"]["old_string"] == "before"
                    assert conn.execute(
                        "SELECT f.old_string,b.session_id FROM file_edits f "
                        "JOIN blocks b ON b.block_id=f.tool_use_block_id WHERE f.session_id=?",
                        (child_id,),
                    ).fetchone()[:] == ("before", child_id)
                    assert (
                        conn.execute(
                            "SELECT old_string FROM file_edits WHERE session_id='codex-session:parent'"
                        ).fetchone()[0]
                        == "after"
                    )
                    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
                else:
                    assert extras["web_constructs"][0]["url"] == "before"
            finally:
                conn.close()

    def test_witness_mismatch_keeps_the_childs_own_content(self, tmp_path: Path) -> None:
        """A same-id, different-content candidate is not this child's branch point.

        ``gp`` carries native id ``m1`` with different content. Re-parsing
        ``parent`` onto ``gp`` deletes the child's anchor row ``parent:n:m1``,
        and the only exact-suffix candidate left is ``gp:n:m1`` -- a different
        message. The child keeps the prefix its own bytes replayed.
        """
        db = tmp_path / "index.db"
        cursor = CursorStore(db)
        conn = _connect(db)
        try:
            write_fixture_index_session(conn, _session("gp", [("m0", "m0"), ("m1", "other-m1")]))
            write_fixture_index_session(conn, _session("parent", [("m0", "m0"), ("m1", "m1"), ("m2", "m2")]))
            child_id = write_fixture_index_session(
                conn, _session("child", [("m0", "m0"), ("m1", "m1"), ("x", "x")], parent="parent")
            )
            conn.commit()
            anchored_id, _witness = _edge(conn, child_id)
            assert anchored_id == "codex-session:parent:n:m1"
            assert _composed(conn, child_id) == (["m0", "m1", "x"], True, None)

            write_fixture_index_session(
                conn, _session("parent", [("m0", "m0"), ("m1", "other-m1"), ("m2", "m2")], parent="gp")
            )
            conn.commit()

            edge = conn.execute(
                "SELECT inheritance, branch_point_message_id FROM session_links WHERE src_session_id = ?",
                (child_id,),
            ).fetchone()
            assert tuple(edge) == ("spawned-fresh", None)
            assert count_dangling_prefix_branch_points(conn) == (0, 0)
            assert cursor.list_convergence_debt() == []
            assert _composed(conn, child_id) == (["m0", "m1", "x"], True, None)
        finally:
            conn.close()

    def test_witness_agreement_still_repairs(self, tmp_path: Path) -> None:
        """Opposite direction: refusing every re-anchor would fail here.

        The same relocation with *matching* content is the repairable half: the
        branch-point message survives in the parent's composed transcript, so
        the edge is re-resolved in-write and nothing is stranded.
        """
        db = tmp_path / "index.db"
        cursor = CursorStore(db)
        conn = _connect(db)
        try:
            write_fixture_index_session(conn, _session("gp", [("m0", "m0"), ("m1", "m1")]))
            write_fixture_index_session(conn, _session("parent", [("m0", "m0"), ("m1", "m1"), ("m2", "m2")]))
            child_id = write_fixture_index_session(
                conn, _session("child", [("m0", "m0"), ("m1", "m1"), ("x", "x")], parent="parent")
            )
            conn.commit()

            write_fixture_index_session(
                conn, _session("parent", [("m0", "m0"), ("m1", "m1"), ("m2", "m2")], parent="gp")
            )
            conn.commit()

            anchored_id, _witness = _edge(conn, child_id)
            assert anchored_id == "codex-session:gp:n:m1"
            assert count_dangling_prefix_branch_points(conn) == (0, 0)
            assert cursor.list_convergence_debt() == []
            assert _composed(conn, child_id) == (["m0", "m1", "x"], True, None)
        finally:
            conn.close()


def test_initial_child_semantic_difference_is_never_discarded_as_a_parent_prefix(tmp_path: Path) -> None:
    """A complete witness must authorize alignment before any child row is dropped."""
    from polylogue.core.enums import Origin
    from polylogue.pipeline.ids import message_semantic_content_address
    from polylogue.sources.tool_outcomes import derive_tool_outcomes

    for kind in ("metadata", "file_edit", "web_constructs"):
        root = tmp_path / kind
        root.mkdir()
        connection = _connect(root / "index.db")
        try:
            parent = _witness_session("parent", extra_kind=kind, extra_value="parent")
            child = _witness_session("child", extra_kind=kind, extra_value="child", parent="parent")
            write_fixture_index_session(connection, parent)
            child_id = write_fixture_index_session(connection, child)
            connection.commit()
            edge = connection.execute(
                "SELECT branch_point_message_id FROM session_links WHERE src_session_id = ?", (child_id,)
            ).fetchone()
            assert edge[0] == (None if kind == "file_edit" else "codex-session:parent:n:m0")
            row = connection.execute(
                "SELECT content_address FROM messages WHERE session_id = ? AND native_id = 'm1'", (child_id,)
            ).fetchone()
            # The writer stores the witness of the canonically normalized
            # message (derived tool outcomes included), the same operand the
            # parent's rows and the child's alignment use.
            stored_child = derive_tool_outcomes(child.messages, child.session_events, origin=Origin.CODEX_SESSION)
            stored_parent = derive_tool_outcomes(parent.messages, parent.session_events, origin=Origin.CODEX_SESSION)
            assert bytes(row[0]) == message_semantic_content_address(stored_child[1])
            assert bytes(row[0]) != message_semantic_content_address(stored_parent[1])
            assert _composed(connection, child_id) == (["m0", "m1", "x"], True, None)
            if kind == "file_edit":
                assert [
                    tuple(row)
                    for row in connection.execute(
                        "SELECT f.session_id, f.old_string, b.session_id FROM file_edits AS f "
                        "JOIN blocks AS b ON b.block_id=f.tool_use_block_id ORDER BY f.session_id"
                    )
                ] == [(child_id, "child", child_id), ("codex-session:parent", "parent", "codex-session:parent")]
                assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        finally:
            connection.close()


@pytest.mark.parametrize("disk", (False, True))
@pytest.mark.parametrize("child_first", (False, True))
@pytest.mark.parametrize("reply_first", (False, True))
def test_file_edit_dependency_boundary_preserves_parent_and_child_edits(
    tmp_path: Path, disk: bool, child_first: bool, reply_first: bool
) -> None:
    def session(native_id: str, *, parent: str | None = None) -> ParsedSession:
        uses = [
            ParsedMessage(
                provider_message_id=f"use-{name}",
                role=Role.ASSISTANT,
                blocks=[ParsedContentBlock(type=BlockType.TOOL_USE, tool_id=name, tool_name="Edit", tool_input={})],
            )
            for name in ("a", "b")
        ]
        replies = [
            ParsedMessage(
                provider_message_id=f"result-{name}",
                parent_message_provider_id=f"use-{name}",
                role=Role.TOOL,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_RESULT,
                        tool_id=name,
                        is_error=False,
                        file_edit=ParsedFileEdit(
                            file_path=f"/fixture/{name}.py",
                            old_string=native_id if name == "b" else "same-a",
                            new_string="after",
                        ),
                    )
                ],
            )
            for name in ("a", "b")
        ]
        messages = [*replies, *uses] if reply_first else [*uses, *replies]
        if parent is not None:
            messages.append(_message("tail", "child tail", len(messages)))
        result = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id,
            parent_session_provider_id=parent,
            branch_type=BranchType.FORK if parent is not None else None,
            messages=messages,
        )
        if not disk:
            return result
        store = SqliteMessageStore(tmp_path / f"{native_id}-messages.db")
        sink = store.new_sink()
        sink.extend(result.messages)
        sink.normalized_messages(result.session_events, origin=Origin.CODEX_SESSION)
        store.conn.commit()
        store.close()
        sealed = SqliteMessageSink(store.path, sink.session_ordinal, count=len(sink))
        return result.model_copy(update={"messages": sealed, "content_hash": str(session_content_hash(result))})

    conn = _connect(tmp_path / "index.db")
    try:
        parent, child = session("parent"), session("child", parent="parent")
        for current in (child, parent) if child_first else (parent, child):
            write_fixture_index_session(conn, current)
        assert [
            tuple(row)
            for row in conn.execute(
                "SELECT f.session_id, b.session_id, b.tool_id, f.old_string FROM file_edits f "
                "JOIN blocks b ON b.block_id=f.tool_use_block_id ORDER BY f.session_id,b.tool_id"
            )
        ] == [
            ("codex-session:child", "codex-session:child", "a", "same-a"),
            ("codex-session:child", "codex-session:child", "b", "child"),
            ("codex-session:parent", "codex-session:parent", "a", "same-a"),
            ("codex-session:parent", "codex-session:parent", "b", "parent"),
        ]
        assert conn.execute(
            "SELECT inheritance,branch_point_message_id FROM session_links WHERE src_session_id='codex-session:child'"
        ).fetchone()[:] == ("spawned-fresh", None)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()
