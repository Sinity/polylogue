"""Compact seed-relative lineage graph (polylogue-4ts.9).

A lineage lookup already told a caller which sessions are related. These tests
pin what the compact relation adds: the seed-relative role of every node and
edge, the branch semantics carried on the edge, stable independent paging that
keeps unresolved and quarantined edges visible, accounting that either matches
composition or says ``unknown``, and an execution that never reads a message
body.

Anti-vacuity: ``test_compact_execution_reads_no_message_body`` records every
statement the derivation issues. A derivation that composed transcripts to
answer these questions -- the shape this relation exists to replace -- is red
there while every other assertion here stays green.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

from polylogue.analysis.lineage_graph import (
    LineageAccountingStatus,
    LineageEdgeResolution,
    LineageEdgeRole,
    LineageNodeRole,
)
from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.derived.lineage.compact import derive_compact_lineage
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

_PARENT = "codex-session:lineage-parent"
_FORK = "codex-session:lineage-fork"
_SPAWNED = "codex-session:lineage-spawned"


def _message(native_id: str, role: Role, text: str) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=native_id,
        role=role,
        text=text,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _seed_family(conn: sqlite3.Connection) -> None:
    """A parent with one prefix-sharing fork and one spawned-fresh subagent."""
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="lineage-parent",
        title="parent",
        messages=[
            _message("p0", Role.USER, "hello"),
            _message("p1", Role.ASSISTANT, "hi there"),
            _message("p2", Role.USER, "keep going"),
        ],
    )
    write_parsed_session_to_archive(conn, parent)
    fork = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="lineage-fork",
        title="fork",
        parent_session_provider_id="lineage-parent",
        branch_type=BranchType.FORK,
        messages=[
            _message("p0", Role.USER, "hello"),
            _message("p1", Role.ASSISTANT, "hi there"),
            _message("f2", Role.USER, "fork diverges"),
            _message("f3", Role.ASSISTANT, "fork reply"),
        ],
    )
    write_parsed_session_to_archive(conn, fork)
    spawned = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="lineage-spawned",
        title="spawned subagent",
        parent_session_provider_id="lineage-parent",
        branch_type=BranchType.SUBAGENT,
        messages=[
            _message("s0", Role.USER, "focused brief"),
            _message("s1", Role.ASSISTANT, "focused answer"),
        ],
    )
    write_parsed_session_to_archive(conn, spawned)
    conn.commit()


def test_seed_relative_roles_and_branch_semantics(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    _seed_family(conn)

    graph = derive_compact_lineage(conn, _FORK, node_limit=None, edge_limit=None)

    assert graph is not None
    roles = {str(node.session_id): node.role for node in graph.nodes}
    assert roles[_FORK] is LineageNodeRole.SEED
    assert roles[_PARENT] is LineageNodeRole.ANCESTOR
    assert roles[_SPAWNED] is LineageNodeRole.SIBLING
    assert str(graph.root_id) == _PARENT

    depths = {str(node.session_id): node.depth_from_seed for node in graph.nodes}
    assert depths[_FORK] == 0
    assert depths[_PARENT] == -1

    seed_parent_edges = [edge for edge in graph.edges if edge.role is LineageEdgeRole.SEED_PARENT]
    assert [str(edge.parent_id) for edge in seed_parent_edges] == [_PARENT]
    fork_edge = seed_parent_edges[0]
    assert fork_edge.inheritance == "prefix-sharing"
    assert fork_edge.branch_point_message_id is not None
    assert fork_edge.resolution is LineageEdgeResolution.RESOLVED
    assert fork_edge.link_type == BranchType.FORK.value

    spawned_edge = next(edge for edge in graph.edges if str(edge.child_id) == _SPAWNED)
    assert spawned_edge.inheritance == "spawned-fresh"
    assert spawned_edge.link_type == BranchType.SUBAGENT.value


def test_spawned_fresh_and_prefix_sharing_accounting_matches_composition(tmp_path: Path) -> None:
    """Composition is the reference; accounting reproduces its size or abstains."""
    from polylogue.storage.sqlite.archive_tiers.write import read_archive_session_envelope

    conn = _connect(tmp_path / "index.db")
    _seed_family(conn)

    graph = derive_compact_lineage(conn, _PARENT, node_limit=None, edge_limit=None)
    assert graph is not None
    accounting = {str(node.session_id): node.accounting for node in graph.nodes}

    for session_id in (_PARENT, _FORK, _SPAWNED):
        entry = accounting[session_id]
        assert entry.status is LineageAccountingStatus.KNOWN
        composed = len(read_archive_session_envelope(conn, session_id).messages)
        assert entry.composed == composed, session_id

    # The fork's own rows are its divergent tail; the rest is the parent's.
    assert accounting[_FORK].inherited == 2
    assert accounting[_FORK].unique == 2
    # A spawned-fresh child inherits nothing, so all of its messages are its own.
    assert accounting[_SPAWNED].inherited == 0
    assert accounting[_SPAWNED].unique == 2


def test_dangling_branch_point_reports_unknown_rather_than_a_number(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    _seed_family(conn)
    conn.execute(
        "UPDATE session_links SET branch_point_message_id = ? WHERE src_session_id = ?",
        (f"{_PARENT}:n:absent", _FORK),
    )
    conn.commit()

    graph = derive_compact_lineage(conn, _FORK, node_limit=None, edge_limit=None)

    assert graph is not None
    fork_accounting = next(node.accounting for node in graph.nodes if str(node.session_id) == _FORK)
    assert fork_accounting.status is LineageAccountingStatus.UNKNOWN
    assert fork_accounting.unique is None
    assert fork_accounting.inherited is None


def test_seed_is_present_on_every_node_page(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    _seed_family(conn)

    first = derive_compact_lineage(conn, _FORK, node_limit=2, node_offset=0)
    second = derive_compact_lineage(conn, _FORK, node_limit=2, node_offset=1)
    last = derive_compact_lineage(conn, _FORK, node_limit=1, node_offset=99)

    for page in (first, second, last):
        assert page is not None
        assert page.seed_node().is_seed
        assert str(page.seed_node().session_id) == _FORK
    assert first is not None
    assert last is not None
    assert first.node_page.total == 2
    assert first.node_page.returned == 1
    assert first.node_page.has_more
    assert not last.node_page.has_more


def test_pagination_is_stable_and_keeps_unresolved_and_quarantined_edges(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    _seed_family(conn)
    conn.execute(
        """
        INSERT INTO session_links
            (src_session_id, dst_origin, dst_native_id, link_type, resolved_dst_session_id,
             inheritance, status, method, confidence, evidence_json, observed_at_ms)
        VALUES (?, 'codex-session', 'never-imported', 'continuation', NULL,
                'prefix-sharing', NULL, 'parser-inference', 0.5, '[]', 0)
        """,
        (_FORK,),
    )
    conn.execute(
        """
        INSERT INTO session_links
            (src_session_id, dst_origin, dst_native_id, link_type, resolved_dst_session_id,
             inheritance, status, method, confidence, evidence_json, observed_at_ms)
        VALUES (?, 'codex-session', 'lineage-parent', 'sidechain', ?,
                'prefix-sharing', 'quarantined', 'contradicted-by-hook-evidence', 0.2, '[]', 0)
        """,
        (_FORK, _PARENT),
    )
    conn.commit()

    full = derive_compact_lineage(conn, _FORK, node_limit=None, edge_limit=None)
    assert full is not None
    resolutions = {edge.resolution for edge in full.edges}
    assert LineageEdgeResolution.UNRESOLVED in resolutions
    assert LineageEdgeResolution.QUARANTINED in resolutions

    unresolved = next(edge for edge in full.edges if edge.resolution is LineageEdgeResolution.UNRESOLVED)
    assert unresolved.parent_id is None
    assert unresolved.parent_native_id == "never-imported"
    assert unresolved.confidence == pytest.approx(0.5)
    assert unresolved.method == "parser-inference"

    # Paging the same collection reproduces the same order with no gap or repeat.
    pages: list[str] = []
    for offset in range(0, full.edge_page.total, 2):
        page = derive_compact_lineage(conn, _FORK, edge_limit=2, edge_offset=offset)
        assert page is not None
        pages.extend(f"{edge.child_id}|{edge.link_type}|{edge.parent_native_id}" for edge in page.edges)
    assert pages == [f"{edge.child_id}|{edge.link_type}|{edge.parent_native_id}" for edge in full.edges]


def test_compact_execution_reads_no_message_body(tmp_path: Path) -> None:
    """The relation answers from identity, links and counts, never from content."""
    conn = _connect(tmp_path / "index.db")
    _seed_family(conn)

    statements: list[str] = []

    class _RecordingConnection:
        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        def execute(self, sql: str, parameters: Sequence[object] = ()) -> sqlite3.Cursor:
            statements.append(" ".join(sql.split()).lower())
            return self._inner.execute(sql, parameters)

    graph = derive_compact_lineage(
        _RecordingConnection(conn),  # type: ignore[arg-type]
        _FORK,
        node_limit=None,
        edge_limit=None,
    )

    assert graph is not None
    assert statements
    for sql in statements:
        assert "blocks" not in sql, sql
        assert "search_text" not in sql, sql
        # ``messages`` may only be counted or ordered by, never projected.
        if " from messages" in sql:
            assert "count(*)" in sql or "select position, variant_index" in sql, sql
