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

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    count_dangling_prefix_branch_points,
    read_archive_session_envelope,
)
from tests.infra.prepared_session import write_prepared_session


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
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
            write_prepared_session(conn, _session("gp", [("m0", "m0"), ("m1", "other-m1")]))
            write_prepared_session(conn, _session("parent", [("m0", "m0"), ("m1", "m1"), ("m2", "m2")]))
            child_id = write_prepared_session(
                conn, _session("child", [("m0", "m0"), ("m1", "m1"), ("x", "x")], parent="parent")
            )
            conn.commit()
            anchored_id, _witness = _edge(conn, child_id)
            assert anchored_id == "codex-session:parent:n:m1"
            assert _composed(conn, child_id) == (["m0", "m1", "x"], True, None)

            write_prepared_session(
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
            write_prepared_session(conn, _session("gp", [("m0", "m0"), ("m1", "m1")]))
            write_prepared_session(conn, _session("parent", [("m0", "m0"), ("m1", "m1"), ("m2", "m2")]))
            child_id = write_prepared_session(
                conn, _session("child", [("m0", "m0"), ("m1", "m1"), ("x", "x")], parent="parent")
            )
            conn.commit()

            write_prepared_session(conn, _session("parent", [("m0", "m0"), ("m1", "m1"), ("m2", "m2")], parent="gp"))
            conn.commit()

            anchored_id, _witness = _edge(conn, child_id)
            assert anchored_id == "codex-session:gp:n:m1"
            assert count_dangling_prefix_branch_points(conn) == (0, 0)
            assert cursor.list_convergence_debt() == []
            assert _composed(conn, child_id) == (["m0", "m1", "x"], True, None)
        finally:
            conn.close()
