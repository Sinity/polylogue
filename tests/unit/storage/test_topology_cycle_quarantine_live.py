"""Cycle quarantine through the LIVE write path (polylogue-4ts.10).

``session_links.status``/``.method`` were declared (``TopologyEdgeStatus``,
``storage/sqlite/archive_tiers/index.py``) but never written by the sole
production writer (``write_parsed_session_to_archive`` ->
``_resolve_session_graph`` / ``_resolve_outbound_session_links``,
``storage/sqlite/archive_tiers/write.py``). A real cycle-detection engine
existed only in ``storage/sqlite/queries/session_links.py``, which has zero
production callers -- test-only dead code
(``tests/unit/insights/test_topology_cycle_rejection.py``). Production
instead relied on ``_refresh_session_projection``'s seen-set short-circuit
and ``_composed_db_signatures``' visited-set truncation to avoid infinite
recursion on a real cycle, silently picking an arbitrary root/branch point
with no persisted evidence of the rejected edge.

These tests reproduce a genuine cross-ingest cycle through
``write_parsed_session_to_archive`` itself (not the dead async engine) and
assert the closing edge is quarantined with evidence, exactly as the two
directly-ported entry points (``_resolve_outbound_session_links`` and the
inbound-parent loop in ``_resolve_session_graph``) are meant to do.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

from devtools.lineage_validation import census_topology_links
from polylogue.archive.message.roles import Role
from polylogue.archive.topology.edge import TopologyEdgeStatus
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _msg(pid: str, role: Role, text: str, position: int) -> ParsedMessage:
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


def _link_row(conn: sqlite3.Connection, src_session_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT status, method, resolved_dst_session_id, evidence_json FROM session_links WHERE src_session_id = ?",
        (src_session_id,),
    ).fetchone()
    assert row is not None
    return cast(sqlite3.Row, row)


def test_cross_ingest_cycle_quarantines_the_closing_edge(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    conn = _connect(db)

    # 1. A lands first with no parent -- it is its own root.
    session_a_v1 = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="A",
        title="A",
        messages=[_msg("a0", Role.USER, "start", 0)],
    )
    a_id = write_parsed_session_to_archive(conn, session_a_v1)
    assert conn.execute("SELECT parent_session_id FROM sessions WHERE session_id = ?", (a_id,)).fetchone()[0] is None

    # 2. B lands claiming A as its parent. A already exists, so this resolves
    # immediately and B.parent_session_id is projected to A.
    session_b = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="B",
        title="B",
        parent_session_provider_id="A",
        messages=[_msg("b0", Role.USER, "child of A", 0)],
    )
    b_id = write_parsed_session_to_archive(conn, session_b)
    assert conn.execute("SELECT parent_session_id FROM sessions WHERE session_id = ?", (b_id,)).fetchone()[0] == a_id

    # 3. A is re-ingested (a corrupted/scrambled export re-asserting lineage)
    # now claiming B as ITS parent -- closing a two-node cycle A -> B -> A.
    session_a_v2 = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="A",
        title="A",
        parent_session_provider_id="B",
        messages=[_msg("a0", Role.USER, "start", 0), _msg("a1", Role.ASSISTANT, "revised", 1)],
    )
    write_parsed_session_to_archive(conn, session_a_v2, force_replace=True)

    # The closing edge (A -> B) must be quarantined, not silently resolved.
    link = _link_row(conn, a_id)
    assert link["status"] == TopologyEdgeStatus.QUARANTINED.value
    assert link["resolved_dst_session_id"] is None
    evidence = json.loads(link["evidence_json"])
    assert evidence["reason"] == "cycle_rejected"
    assert a_id in evidence["cycle_path"]
    assert b_id in evidence["cycle_path"]

    # A's parent_session_id fast-path projection must stay NULL -- the
    # composition/ancestry walk must never enter the cycle.
    assert conn.execute("SELECT parent_session_id FROM sessions WHERE session_id = ?", (a_id,)).fetchone()[0] is None

    # B's own (earlier, legitimate) edge to A is untouched by A's rejected edge.
    b_link = _link_row(conn, b_id)
    assert b_link["status"] is None
    assert b_link["resolved_dst_session_id"] == a_id

    census = census_topology_links(conn, sample_unresolved=0)
    assert census["checked"] is True
    assert census["empty_effective_status_count"] == 0
    assert census["empty_method_count"] == 0
    assert census["effective_status_counts"] == {"quarantined": 1, "resolved": 1}
    assert census["cycle_evidence_count"] == 1

    # Anti-vacuity: the census must observe a production-row mutation rather
    # than merely restating the expected fixture shape.
    conn.execute("UPDATE session_links SET method = '' WHERE src_session_id = ?", (b_id,))
    mutated = census_topology_links(conn, sample_unresolved=0)
    assert mutated["empty_method_count"] == 1


def test_self_referential_edge_quarantines_without_touching_projection(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    conn = _connect(db)

    session_v1 = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="self-loop",
        title="self-loop",
        messages=[_msg("s0", Role.USER, "start", 0)],
    )
    session_id = write_parsed_session_to_archive(conn, session_v1)

    session_v2 = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="self-loop",
        title="self-loop",
        parent_session_provider_id="self-loop",
        messages=[_msg("s0", Role.USER, "start", 0), _msg("s1", Role.ASSISTANT, "claims itself as parent", 1)],
    )
    write_parsed_session_to_archive(conn, session_v2, force_replace=True)

    link = _link_row(conn, session_id)
    assert link["status"] == TopologyEdgeStatus.QUARANTINED.value
    evidence = json.loads(link["evidence_json"])
    assert evidence["cycle_path"] == [session_id, session_id]
    assert (
        conn.execute("SELECT parent_session_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0] is None
    )


def test_diamond_dag_is_not_mistaken_for_a_cycle(tmp_path: Path) -> None:
    """B -> D and C -> D (both children of D) is a legitimate shared-parent
    shape, not a cycle -- the resolver must resolve both edges cleanly."""
    db = tmp_path / "index.db"
    conn = _connect(db)

    session_d = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="D",
        title="D",
        messages=[_msg("d0", Role.USER, "root", 0)],
    )
    d_id = write_parsed_session_to_archive(conn, session_d)

    session_b = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="B",
        title="B",
        parent_session_provider_id="D",
        messages=[_msg("b0", Role.USER, "child of D", 0)],
    )
    b_id = write_parsed_session_to_archive(conn, session_b)

    session_c = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="C",
        title="C",
        parent_session_provider_id="D",
        messages=[_msg("c0", Role.USER, "also child of D", 0)],
    )
    c_id = write_parsed_session_to_archive(conn, session_c)

    for child_id in (b_id, c_id):
        link = _link_row(conn, child_id)
        assert link["status"] is None
        assert link["resolved_dst_session_id"] == d_id
        assert (
            conn.execute("SELECT parent_session_id FROM sessions WHERE session_id = ?", (child_id,)).fetchone()[0]
            == d_id
        )
