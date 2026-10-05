"""Citation anchors agree with production lineage reads; topology describes its own component."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.block_anchor import BlockAnchor, resolve_block_anchor
from polylogue.storage.derived.topology.derivation import TopologyNodeInput, compose_session_topology
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import read_archive_session_envelope
from tests.infra.index_writer import write_fixture_index_session


def _connection(path: Path | str) -> sqlite3.Connection:
    conn = connect_measured(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _session(conn: sqlite3.Connection, native: str, texts: list[tuple[str, str]]) -> str:
    return write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native,
            messages=[
                ParsedMessage(
                    provider_message_id=message_id,
                    role=Role.ASSISTANT,
                    material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                )
                for message_id, text in texts
            ],
        ),
    )


def _anchor(conn: sqlite3.Connection, session_id: str) -> BlockAnchor:
    row = conn.execute(
        "SELECT b.message_id, b.content_hash FROM blocks b JOIN messages m ON m.message_id = b.message_id "
        "WHERE m.session_id = ? ORDER BY m.position, b.position LIMIT 1",
        (session_id,),
    ).fetchone()
    assert row is not None
    return BlockAnchor(session_id, str(row["message_id"]), bytes(row["content_hash"]).hex())


def _link(
    conn: sqlite3.Connection, child: str, parent: str, inheritance: str | None, branch: str | None = None
) -> None:
    native = conn.execute("SELECT native_id FROM sessions WHERE session_id = ?", (parent,)).fetchone()[0]
    conn.execute(
        "INSERT INTO session_links(src_session_id, dst_origin, dst_native_id, link_type, "
        "resolved_dst_session_id, branch_point_message_id, inheritance, status, confidence, "
        "evidence_json, observed_at_ms) "
        "VALUES (?, 'codex-session', ?, 'fork', ?, ?, ?, NULL, 1.0, '[]', 0)",
        (child, native, parent, branch, inheritance),
    )
    conn.commit()


def test_anchor_does_not_relocate_through_an_unclassified_edge(tmp_path: Path) -> None:
    """A resolved parent with NULL inheritance previously supplied unrelated evidence."""
    with closing(_connection(tmp_path / "index.db")) as conn:
        parent = _session(conn, "parent", [("p", "unique evidence")])
        child = _session(conn, "child", [])
        original = _anchor(conn, parent)
        _link(conn, child, parent, None)
        result = resolve_block_anchor(conn, BlockAnchor(child, original.message_id, original.content_hash_hex))
        assert result.state == "missing"


def test_anchor_relocation_reaches_an_ancestor_beyond_the_former_search_cap(tmp_path: Path) -> None:
    """The anchor's own 512-node search cap hid a block every production read exposes.

    Production composition has no depth cap. With the block only in a root
    520 prefix-sharing links up, the capped neighbourhood never reached it and
    reported ``missing``; the relocation must name a session whose read
    really contains the block. The 521-session chain uses its declared Index:
    per-session commits, not the resolver, dominate the test's cost.
    """
    with closing(_connection(tmp_path / "index.db")) as conn:
        root = _session(conn, "depth-0", [("m", "root evidence")])
        original = _anchor(conn, root)
        parent = root
        for depth in range(1, 521):
            child = _session(conn, f"depth-{depth}", [("m", f"tail {depth}")])
            _link(conn, child, parent, "prefix-sharing", _anchor(conn, parent).message_id)
            parent = child
        result = resolve_block_anchor(conn, BlockAnchor(parent, original.message_id, original.content_hash_hex))
        assert result.state == "relocated_lineage", result.detail
        assert result.resolved_message_id == original.message_id
        resolved_view = result.detail.split("composed lineage session ", 1)[1].split(" via ", 1)[0]
        envelope = read_archive_session_envelope(conn, resolved_view)
        assert original.message_id in {message.message_id for message in envelope.messages}
        leaf = read_archive_session_envelope(conn, parent)
        assert original.message_id in {message.message_id for message in leaf.messages}


def test_anchor_hash_lookup_does_not_bind_every_transcript_message(tmp_path: Path) -> None:
    """The previous IN list exceeds the connection's real SQLite bind-variable limit."""
    with closing(_connection(tmp_path / "index.db")) as conn:
        parent = _session(conn, "many", [(f"m{i}", f"evidence {i}") for i in range(40)])
        child = _session(conn, "tail", [])
        original = _anchor(conn, parent)
        branch = conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position DESC LIMIT 1",
            (parent,),
        ).fetchone()[0]
        _link(conn, child, parent, "prefix-sharing", str(branch))
        previous = conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 12)
        try:
            result = resolve_block_anchor(conn, BlockAnchor(child, original.message_id, original.content_hash_hex))
        finally:
            conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous)
        assert result.state == "relocated_lineage"
        assert result.resolved_message_id == original.message_id


def test_anchor_relocates_locally_after_its_original_message_disappears(tmp_path: Path) -> None:
    """The old missing-message branch bypassed same-session hash relocation."""
    with closing(_connection(tmp_path / "index.db")) as conn:
        session_id = _session(conn, "drift", [("old", "unchanged evidence")])
        original = _anchor(conn, session_id)
        _session(conn, "drift", [("new", "unchanged evidence")])
        conn.execute("DELETE FROM messages WHERE message_id = ?", (original.message_id,))
        conn.commit()
        assert conn.execute("SELECT 1 FROM messages WHERE message_id = ?", (original.message_id,)).fetchone() is None
        replacement = _anchor(conn, session_id)
        result = resolve_block_anchor(conn, original)
        assert result.state == "drifted_message"
        assert result.resolved_message_id == replacement.message_id


def test_anchor_refuses_two_distinct_sibling_hash_matches(tmp_path: Path) -> None:
    """Returning on the first neighbour formerly concealed a second physical candidate."""
    with closing(_connection(tmp_path / "index.db")) as conn:
        parent = _session(conn, "family-root", [])
        target = _session(conn, "target", [])
        first = _session(conn, "sibling-a", [("m", "shared text")])
        second = _session(conn, "sibling-b", [("m", "shared text")])
        for child in (target, first, second):
            _link(conn, child, parent, "spawned-fresh")
        first_anchor = _anchor(conn, first)
        second_anchor = _anchor(conn, second)
        result = resolve_block_anchor(conn, BlockAnchor(target, "gone", first_anchor.content_hash_hex))
        assert result.state == "ambiguous"
        assert set(result.candidates) == {(first_anchor.message_id, 0), (second_anchor.message_id, 0)}


def _edge(child: str, parent: str, kind: str = "fork") -> dict[str, object]:
    return {
        "src_session_id": child,
        "dst_origin": "codex-session",
        "dst_native_id": parent,
        "resolved_dst_session_id": parent,
        "link_type": kind,
        "inheritance": "spawned-fresh",
        "status": None,
        "confidence": 1.0,
        "evidence_json": "[]",
    }


def test_topology_siblings_are_nodes_not_duplicate_evidence_rows() -> None:
    """Two accepted assertions for b formerly yielded b twice in a's siblings."""
    nodes = [TopologyNodeInput(name, "codex-session") for name in ("root", "a", "b")]
    links = [_edge("a", "root"), _edge("b", "root"), _edge("b", "root", "continuation")]
    graph = compose_session_topology("a", nodes, links)
    assert graph is not None
    assert tuple(map(str, graph.siblings("a"))) == ("b",)
    assert len(graph.edges) == 3, "structural deduplication must retain every evidence assertion"


def test_topology_fault_flags_are_scoped_to_the_returned_component() -> None:
    """An unrelated cycle/conflict formerly marked this healthy component faulty."""
    names = ("root", "good", "cycle-a", "cycle-b", "conflict", "parent-a", "parent-b")
    nodes = [TopologyNodeInput(name, "codex-session") for name in names]
    links = [
        _edge("good", "root"),
        _edge("cycle-a", "cycle-b"),
        _edge("cycle-b", "cycle-a"),
        _edge("conflict", "parent-a"),
        _edge("conflict", "parent-b", "continuation"),
    ]
    healthy = compose_session_topology("good", nodes, links)
    assert healthy is not None
    assert not healthy.cycle_detected and not healthy.conflicting_parent_detected
    cyclic = compose_session_topology("cycle-a", nodes, links)
    conflicting = compose_session_topology("conflict", nodes, links)
    assert cyclic is not None and cyclic.cycle_detected
    assert conflicting is not None and conflicting.conflicting_parent_detected
