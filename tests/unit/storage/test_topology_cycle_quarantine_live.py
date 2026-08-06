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

import aiosqlite
import pytest

from devtools.lineage_validation import census_topology_links
from polylogue.archive.message.roles import Role
from polylogue.archive.topology.edge import TopologyEdgeStatus
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import read_archive_session_envelope, write_parsed_session_to_archive
from polylogue.storage.sqlite.queries.message_query_reads import get_messages


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


@pytest.mark.asyncio
async def test_cross_ingest_cycle_quarantines_the_closing_edge_without_losing_prefix(tmp_path: Path) -> None:
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
        messages=[
            _msg("a-copy-b0", Role.USER, "child of A", 0),
            _msg("a1", Role.ASSISTANT, "revised", 1),
        ],
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

    # Anti-vacuity: the first A message exactly matches B's full stored
    # transcript. Without the pre-normalization cycle check, the writer slices
    # it as an inherited prefix before quarantining A -> B, and both production
    # readers then serve only the second message.
    own_message_ids = [
        str(row[0])
        for row in conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position, variant_index",
            (a_id,),
        ).fetchall()
    ]
    assert len(own_message_ids) == 2
    quarantined_envelope = read_archive_session_envelope(conn, a_id)
    assert [message.message_id for message in quarantined_envelope.messages] == own_message_ids
    async with aiosqlite.connect(db) as reader:
        reader.row_factory = sqlite3.Row
        async_messages = await get_messages(reader, a_id)
    assert [message.message_id for message in async_messages] == own_message_ids

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
    assert census["malformed_quarantine_evidence_count"] == 0
    assert census["budget_exhausted_quarantine_evidence_count"] == 0
    assert census["quarantined_with_resolved_parent_count"] == 0
    assert census["quarantined_with_stale_projection_count"] == 0

    valid_evidence = link["evidence_json"]
    conn.execute("UPDATE session_links SET evidence_json = '{malformed' WHERE src_session_id = ?", (a_id,))
    malformed = census_topology_links(conn, sample_unresolved=0)
    assert malformed["cycle_evidence_count"] == 0
    assert malformed["malformed_quarantine_evidence_count"] == 1
    assert malformed["quarantined_without_cycle_evidence"] == 1

    unrelated_evidence = json.dumps(
        {
            "reason": "cycle_rejected",
            "cycle_path": ["unrelated-a", "unrelated-b"],
            "detected_at_ms": 1,
        }
    )
    conn.execute(
        "UPDATE session_links SET evidence_json = ? WHERE src_session_id = ?",
        (unrelated_evidence, a_id),
    )
    unrelated = census_topology_links(conn, sample_unresolved=0)
    assert unrelated["cycle_evidence_count"] == 0
    assert unrelated["malformed_quarantine_evidence_count"] == 1
    assert unrelated["budget_exhausted_quarantine_evidence_count"] == 0

    fabricated_cycle_evidence = json.dumps(
        {
            "reason": "cycle_rejected",
            "cycle_path": [a_id, b_id, a_id],
            "detected_at_ms": 1,
        }
    )
    conn.execute("UPDATE sessions SET parent_session_id = NULL WHERE session_id = ?", (b_id,))
    conn.execute(
        "UPDATE session_links SET evidence_json = ? WHERE src_session_id = ?",
        (fabricated_cycle_evidence, a_id),
    )
    fabricated = census_topology_links(conn, sample_unresolved=0)
    assert fabricated["cycle_evidence_count"] == 0
    assert fabricated["malformed_quarantine_evidence_count"] == 1
    assert fabricated["budget_exhausted_quarantine_evidence_count"] == 0
    assert fabricated["quarantined_without_cycle_evidence"] == 1
    conn.execute("UPDATE sessions SET parent_session_id = ? WHERE session_id = ?", (a_id, b_id))

    parent_message_id = conn.execute(
        "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position LIMIT 1", (b_id,)
    ).fetchone()[0]
    conn.execute(
        """
        UPDATE session_links
        SET evidence_json = ?, resolved_dst_session_id = ?, branch_point_message_id = ?,
            inheritance = 'prefix-sharing'
        WHERE src_session_id = ?
        """,
        (valid_evidence, b_id, parent_message_id, a_id),
    )
    conn.execute(
        "UPDATE sessions SET parent_session_id = ?, root_session_id = ? WHERE session_id = ?",
        (b_id, b_id, a_id),
    )
    quarantined_read = read_archive_session_envelope(conn, a_id)
    assert quarantined_read.lineage_inheritance == "none"
    assert [message.message_id for message in quarantined_read.messages] == own_message_ids
    contradictory = census_topology_links(conn, sample_unresolved=0)
    assert contradictory["quarantined_with_resolved_parent_count"] == 1
    assert contradictory["quarantined_with_stale_projection_count"] == 1
    assert contradictory["cycle_evidence_count"] == 1

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


def test_over_budget_acyclic_walk_is_not_recorded_as_a_cycle_and_keeps_prefix(tmp_path: Path) -> None:
    """The live writer must distinguish an indeterminate deep walk from a cycle.

    Production dependencies: pre-slice cycle classification, outbound link
    quarantine, and the synchronous composed reader. Mutation: returning a
    cycle path at the walk budget records `cycle_rejected`; treating exhaustion
    as acyclic slices the copied parent prefix and serves only the tail.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="deep-0",
        title="deep parent",
        messages=[_msg("p0", Role.USER, "copied parent prefix", 0)],
    )
    parent_id = write_parsed_session_to_archive(conn, parent)
    child_v1 = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="deep-child",
        title="deep child",
        messages=[_msg("c0", Role.USER, "original child", 0)],
    )
    child_id = write_parsed_session_to_archive(conn, child_v1)

    for position in range(1024, 0, -1):
        native_id = f"deep-{position}"
        parent_session_id = None if position == 1024 else f"codex-session:deep-{position + 1}"
        conn.execute(
            """
            INSERT INTO sessions(native_id, origin, parent_session_id, content_hash)
            VALUES (?, 'codex-session', ?, ?)
            """,
            (native_id, parent_session_id, bytes(32)),
        )
    conn.execute(
        "UPDATE sessions SET parent_session_id = ? WHERE session_id = ?",
        ("codex-session:deep-1", parent_id),
    )

    child_v2 = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="deep-child",
        title="deep child",
        parent_session_provider_id="deep-0",
        messages=[
            _msg("copy-p0", Role.USER, "copied parent prefix", 0),
            _msg("c1", Role.ASSISTANT, "child tail", 1),
        ],
    )
    write_parsed_session_to_archive(conn, child_v2, force_replace=True)

    link = _link_row(conn, child_id)
    evidence = json.loads(link["evidence_json"])
    assert link["status"] == TopologyEdgeStatus.QUARANTINED.value
    assert evidence["reason"] == "cycle_walk_budget_exhausted"
    assert "cycle_path" not in evidence
    assert evidence["walk_budget"] == 1024
    assert len(evidence["walk_path"]) == 1026
    own_message_ids = [
        str(row[0])
        for row in conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position, variant_index",
            (child_id,),
        ).fetchall()
    ]
    assert len(own_message_ids) == 2
    envelope = read_archive_session_envelope(conn, child_id)
    assert [message.message_id for message in envelope.messages] == own_message_ids
    census = census_topology_links(conn, sample_unresolved=0)
    assert census["cycle_evidence_count"] == 0
    assert census["malformed_quarantine_evidence_count"] == 0
    assert census["budget_exhausted_quarantine_evidence_count"] == 1
    assert census["quarantined_without_cycle_evidence"] == 1


def test_quarantined_alternate_parent_does_not_invalidate_resolved_projection(tmp_path: Path) -> None:
    """A valid parent projection can coexist with a rejected alternate edge.

    Production dependencies: repeated writer assertions, cycle quarantine,
    projection refresh, and the topology census. Mutation: counting any parent
    pointer on a child with a quarantined edge reports this valid projection as
    stale even though the child's earlier resolved edge still supports it.
    """
    conn = _connect(tmp_path / "index.db")
    session_a = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="multi-A",
        title="A",
        messages=[_msg("a0", Role.USER, "root A", 0)],
    )
    a_id = write_parsed_session_to_archive(conn, session_a)
    session_b_v1 = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="multi-B",
        title="B",
        parent_session_provider_id="multi-A",
        messages=[_msg("b0", Role.USER, "B follows A", 0)],
    )
    b_id = write_parsed_session_to_archive(conn, session_b_v1)
    session_c = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="multi-C",
        title="C",
        parent_session_provider_id="multi-B",
        messages=[_msg("c0", Role.USER, "C follows B", 0)],
    )
    write_parsed_session_to_archive(conn, session_c)

    session_b_v2 = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="multi-B",
        title="B",
        parent_session_provider_id="multi-C",
        messages=[_msg("b1", Role.USER, "B asserts unsafe alternate C", 0)],
    )
    write_parsed_session_to_archive(conn, session_b_v2, merge_append=True)

    links = conn.execute(
        "SELECT dst_native_id, status FROM session_links WHERE src_session_id = ? ORDER BY dst_native_id",
        (b_id,),
    ).fetchall()
    assert [(row[0], row[1]) for row in links] == [
        ("multi-A", None),
        ("multi-C", TopologyEdgeStatus.QUARANTINED.value),
    ]
    assert conn.execute("SELECT parent_session_id FROM sessions WHERE session_id = ?", (b_id,)).fetchone()[0] == a_id
    census = census_topology_links(conn, sample_unresolved=0)
    assert census["cycle_evidence_count"] == 1
    assert census["quarantined_with_stale_projection_count"] == 0


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
