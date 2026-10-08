"""Hook-evidence authority on the LIVE topology write path.

Codex ``thread_spawn_edges`` reach ``index.db`` as the
thread-state graph projection of the retained state export, but
nothing consumed them for topology: the only artifact was
``context/codex_spawn_edge_correlation.reconcile_codex_spawn_edges``, a
READ-ONLY counter reachable solely through the API facade. It reports
``inferred_only`` / ``authoritative_only`` as set differences over
``(parent, child)`` pairs, so a genuine contradiction -- hook evidence and
transcript inference naming DIFFERENT parents for the SAME child -- shows up
as one entry in each bucket, losing the fact that the two claims compete.
Nothing resolved such a conflict, and nothing stopped a later re-parse from
overwriting an authoritative result.

Two structural facts drive the design under test:

1. ``session_links``' primary key is ``(src_session_id, dst_origin,
   dst_native_id, link_type)``. Contradictory parents therefore do NOT
   overwrite each other -- they land as two coexisting, independently
   resolvable rows. Conflict is consequently scoped to ``(child, link_type)``,
   not to the primary key.
2. ``session_links`` lives in the REBUILDABLE index tier. Only a derivation
   running inside ``write_parsed_session_to_archive`` -- the choke point
   shared by live ingest and full raw replay -- survives a reindex, which is
   why this is a write-path concern rather than a convergence stage.

The losing edge carries ``TopologyEdgeStatus.AUTHORITY_CONTRADICTED``, a
member distinct from ``QUARANTINED``. Both exclude an edge from composition,
but they are different defects and must stay distinguishable forever: a
quarantine is a structural cycle-break (the graph SHAPE is wrong), while an
authority contradiction is a provenance verdict (the shape is fine; this
particular claim was overruled). ``method`` is an unconstrained TEXT column
and cannot carry that distinction safely.

The eight queries that exclude edges from composition previously each
hardcoded ``!= 'quarantined'``. They now read
``topology_status_composes_sql()``, generated from
``COMPOSITION_EXCLUDED_TOPOLOGY_STATUSES``, so a future exclusion class cannot
be silently readmitted by a call site nobody remembered to update.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.topology.edge import (
    HOOK_AUTHORITATIVE_LINK_METHOD,
    HOOK_CONTRADICTED_LINK_METHOD,
    HOOK_SUPERSEDED_LINK_METHOD,
    TopologyEdgeStatus,
)
from polylogue.core.enums import BlockType, LinkType, Origin, Provider
from polylogue.logging import capture
from polylogue.sources.codex_state_projection import write_thread_state_projection
from polylogue.sources.parsers import codex_state
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.write import (
    ConnectionSessionSourceRead,
    prepare_session_write,
    raw_source_path,
)
from tests.infra.archive_templates import bootstrapped_tier_path
from tests.infra.index_writer import write_fixture_index_session
from tests.infra.thread_state import seed_spawn_edges

_CHILD = "child-thread"
_HOOK_PARENT = "hook-parent-thread"
_PARSER_PARENT = "parser-parent-thread"


def _index_conn(path: Path) -> sqlite3.Connection:
    conn = connect_measured(bootstrapped_tier_path(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _source_conn(path: Path) -> sqlite3.Connection:
    conn = connect_measured(bootstrapped_tier_path(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
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


def _session(provider_session_id: str, *, parent: str | None = None) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=provider_session_id,
        title=provider_session_id,
        parent_session_provider_id=parent,
        branch_type=None,
        messages=[_msg(f"{provider_session_id}-0", Role.USER, f"content of {provider_session_id}", 0)],
    )


def _write_spawn_edge(
    conn: sqlite3.Connection, *, parent: str, child: str, observed_at_ms: int = 1_760_000_000_000
) -> None:
    """Project one spawn edge through the production thread-state writer.

    The child-side lookup under test resolves the parent from the child's
    thread id.
    """
    seed_spawn_edges(conn, [(parent, child, "spawned")], observed_at_ms=observed_at_ms)


def _links(conn: sqlite3.Connection, src_session_id: str) -> dict[str, sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT dst_native_id, link_type, status, method, resolved_dst_session_id, evidence_json
        FROM session_links WHERE src_session_id = ?
        """,
        (src_session_id,),
    ).fetchall()
    return {str(row["dst_native_id"]): row for row in rows}


# ---------------------------------------------------------------------------
# Red twin 1: hook evidence wins a contradiction
# ---------------------------------------------------------------------------


def test_contradiction_resolves_to_the_hook_parent(tmp_path: Path) -> None:
    """Hook evidence wins, and the composed parent is the hook's parent.

    Red twin: drop the contradiction branch from ``_write_session_link`` (so
    the parser edge is written unqualified) and the child composes through
    ``_PARSER_PARENT`` instead, because ``_refresh_session_projection`` picks
    the first non-quarantined edge by ``observed_at_ms``.
    """
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    _write_spawn_edge(index, parent=_HOOK_PARENT, child=_CHILD)

    write_fixture_index_session(index, _session(_HOOK_PARENT), source_conn=source)
    write_fixture_index_session(index, _session(_PARSER_PARENT), source_conn=source)
    child_id = write_fixture_index_session(index, _session(_CHILD, parent=_PARSER_PARENT), source_conn=source)

    links = _links(index, child_id)
    assert set(links) == {_HOOK_PARENT, _PARSER_PARENT}, "both evidence sources must be retained"

    winner = links[_HOOK_PARENT]
    assert winner["status"] is None
    assert winner["method"] == HOOK_AUTHORITATIVE_LINK_METHOD
    assert winner["resolved_dst_session_id"] == f"{Origin.CODEX_SESSION.value}:{_HOOK_PARENT}"

    loser = links[_PARSER_PARENT]
    assert loser["status"] == TopologyEdgeStatus.AUTHORITY_CONTRADICTED.value
    assert loser["method"] == HOOK_CONTRADICTED_LINK_METHOD
    assert json.loads(loser["evidence_json"])["codex_thread_spawn_edge_parent"] == _HOOK_PARENT

    # The decision is visible in composition, not merely in the edge rows.
    projected = index.execute("SELECT parent_session_id FROM sessions WHERE session_id = ?", (child_id,)).fetchone()[0]
    assert projected == f"{Origin.CODEX_SESSION.value}:{_HOOK_PARENT}"


# ---------------------------------------------------------------------------
# Red twin 2: the conflict is durable and typed, never a silent overwrite
# ---------------------------------------------------------------------------


def test_reparse_cannot_downgrade_authoritative_evidence(tmp_path: Path) -> None:
    """A later inference-only write must not clobber hook authority.

    This is the concrete hole the guard closes: ``_write_session_link`` used a
    bare ``INSERT OR REPLACE``, which rewrites every column of the matching
    primary key. Re-ingesting the child WITHOUT a source handle (an ordinary
    index-only reprocess) previously reset ``method`` to ``parser-parent`` and
    ``status`` to ``NULL``.

    Red twin: restore ``INSERT OR REPLACE`` in ``_upsert_session_link`` and the
    post-reparse assertions below fail -- ``method`` reverts and the
    contradiction becomes two indistinguishable resolvable rows.
    """
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    _write_spawn_edge(index, parent=_HOOK_PARENT, child=_CHILD)

    write_fixture_index_session(index, _session(_HOOK_PARENT), source_conn=source)
    write_fixture_index_session(index, _session(_PARSER_PARENT), source_conn=source)
    child_id = write_fixture_index_session(index, _session(_CHILD, parent=_PARSER_PARENT), source_conn=source)
    assert _links(index, child_id)[_HOOK_PARENT]["method"] == HOOK_AUTHORITATIVE_LINK_METHOD

    # Re-parse with NO hook evidence available at all.
    write_fixture_index_session(index, _session(_CHILD, parent=_PARSER_PARENT), source_conn=None)

    after = _links(index, child_id)
    assert after[_HOOK_PARENT]["method"] == HOOK_AUTHORITATIVE_LINK_METHOD, (
        "inference-only re-parse downgraded an authoritative edge"
    )
    assert after[_HOOK_PARENT]["status"] is None
    # And the contradiction remains typed and queryable rather than collapsing
    # into two look-alike rows.
    quarantined = index.execute(
        """
        SELECT COUNT(*) FROM session_links
        WHERE src_session_id = ? AND status = ? AND method = ?
        """,
        (child_id, TopologyEdgeStatus.AUTHORITY_CONTRADICTED.value, HOOK_CONTRADICTED_LINK_METHOD),
    ).fetchone()[0]
    assert quarantined == 1


def test_conflict_state_is_queryable_by_typed_status_and_method(tmp_path: Path) -> None:
    """The durable conflict state is discoverable without parsing prose."""
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    _write_spawn_edge(index, parent=_HOOK_PARENT, child=_CHILD)

    write_fixture_index_session(index, _session(_HOOK_PARENT), source_conn=source)
    write_fixture_index_session(index, _session(_PARSER_PARENT), source_conn=source)
    write_fixture_index_session(index, _session(_CHILD, parent=_PARSER_PARENT), source_conn=source)

    conflicts = index.execute(
        """
        SELECT src_session_id, dst_native_id, evidence_json
        FROM session_links
        WHERE status = ? AND method = ?
        """,
        (TopologyEdgeStatus.AUTHORITY_CONTRADICTED.value, HOOK_CONTRADICTED_LINK_METHOD),
    ).fetchall()
    assert len(conflicts) == 1
    evidence = json.loads(conflicts[0]["evidence_json"])
    # Both sides of the disagreement are recoverable from the row itself.
    assert evidence["parent_session_provider_id"] == _PARSER_PARENT
    assert evidence["codex_thread_spawn_edge_parent"] == _HOOK_PARENT


def test_hook_only_edge_is_written_when_inference_found_none(tmp_path: Path) -> None:
    """An authoritative edge transcript inference never found is still recorded."""
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    _write_spawn_edge(index, parent=_HOOK_PARENT, child=_CHILD)

    write_fixture_index_session(index, _session(_HOOK_PARENT), source_conn=source)
    child_id = write_fixture_index_session(index, _session(_CHILD), source_conn=source)

    links = _links(index, child_id)
    assert set(links) == {_HOOK_PARENT}
    assert links[_HOOK_PARENT]["method"] == HOOK_AUTHORITATIVE_LINK_METHOD
    assert links[_HOOK_PARENT]["link_type"] == LinkType.SUBAGENT.value
    assert (
        index.execute("SELECT session_kind FROM sessions WHERE session_id = ?", (child_id,)).fetchone()[0] == "subagent"
    )


def test_hook_only_classification_survives_merge_append(tmp_path: Path) -> None:
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    _write_spawn_edge(index, parent=_HOOK_PARENT, child=_CHILD)

    write_fixture_index_session(index, _session(_HOOK_PARENT), source_conn=source)
    child_id = write_fixture_index_session(index, _session(_CHILD), source_conn=source)
    appended = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=_CHILD,
        title=_CHILD,
        messages=[_msg(f"{_CHILD}-append", Role.USER, "appended content", 1)],
    )
    write_fixture_index_session(index, appended, source_conn=source, merge_append=True)

    assert (
        index.execute("SELECT session_kind FROM sessions WHERE session_id = ?", (child_id,)).fetchone()[0] == "subagent"
    )


def test_agreeing_evidence_upgrades_the_single_edge(tmp_path: Path) -> None:
    """Agreement is not a conflict: one edge, marked authoritative."""
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    _write_spawn_edge(index, parent=_HOOK_PARENT, child=_CHILD)

    write_fixture_index_session(index, _session(_HOOK_PARENT), source_conn=source)
    child_id = write_fixture_index_session(index, _session(_CHILD, parent=_HOOK_PARENT), source_conn=source)

    links = _links(index, child_id)
    assert set(links) == {_HOOK_PARENT}
    assert links[_HOOK_PARENT]["method"] == HOOK_AUTHORITATIVE_LINK_METHOD
    assert links[_HOOK_PARENT]["status"] is None
    quarantined = index.execute("SELECT COUNT(*) FROM session_links WHERE status IS NOT NULL").fetchone()[0]
    assert quarantined == 0


# ---------------------------------------------------------------------------
# Red twin 3: no hook evidence => byte-identical to today
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_unrelated_evidence", [False, True])
def test_no_state_evidence_is_byte_identical_to_the_parser_only_path(
    tmp_path: Path, with_unrelated_evidence: bool
) -> None:
    """Absent evidence is silence, never a conflict.

    Covers both shapes of "no evidence": an empty projection, and a projection
    that simply says nothing about this child. Red twin: mark edges
    unconditionally (drop the ``hook_parent is not None`` guards) and these
    rows stop matching the parser-only baseline.
    """
    baseline_index = _index_conn(tmp_path / "baseline.db")
    write_fixture_index_session(baseline_index, _session(_PARSER_PARENT))
    baseline_child = write_fixture_index_session(baseline_index, _session(_CHILD, parent=_PARSER_PARENT))
    baseline = dict(_links(baseline_index, baseline_child)[_PARSER_PARENT])

    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    if with_unrelated_evidence:
        # Evidence exists, but about a completely unrelated child.
        _write_spawn_edge(index, parent="unrelated-parent", child="unrelated-child")

    write_fixture_index_session(index, _session(_PARSER_PARENT), source_conn=source)
    child_id = write_fixture_index_session(index, _session(_CHILD, parent=_PARSER_PARENT), source_conn=source)
    observed = dict(_links(index, child_id)[_PARSER_PARENT])

    assert observed == baseline
    assert observed["method"] == "parser-parent"
    assert observed["status"] is None


# ---------------------------------------------------------------------------
# Composition exclusion, through the mechanism that actually enforces it
# ---------------------------------------------------------------------------


def test_contradicted_edge_is_excluded_from_composition(tmp_path: Path) -> None:
    """The load-bearing behavioural claim: lineage never composes through a loser.

    This pins the OPERATIVE mechanism -- ``_resolve_outbound_session_links``'
    ``status IS NULL`` gate -- rather than the exclusion set. A cold review
    established that removing ``AUTHORITY_CONTRADICTED`` from
    ``COMPOSITION_EXCLUDED_TOPOLOGY_STATUSES`` changes nothing observable,
    because a contradicted edge never acquires ``resolved_dst_session_id`` in
    the first place; the exclusion set is defense-in-depth for edges that
    resolve BEFORE acquiring a status. So this asserts the real invariant at
    the layer that enforces it: the contradicted edge stays unresolved, and the
    child's composed parent is the hook's parent, never the parser's.
    """
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    _write_spawn_edge(index, parent=_HOOK_PARENT, child=_CHILD)

    write_fixture_index_session(index, _session(_HOOK_PARENT), source_conn=source)
    write_fixture_index_session(index, _session(_PARSER_PARENT), source_conn=source)
    child_id = write_fixture_index_session(index, _session(_CHILD, parent=_PARSER_PARENT), source_conn=source)

    links = _links(index, child_id)
    assert links[_PARSER_PARENT]["resolved_dst_session_id"] is None, (
        "a contradicted edge must never resolve; if it did, composition could traverse it"
    )
    assert links[_HOOK_PARENT]["resolved_dst_session_id"] == f"{Origin.CODEX_SESSION.value}:{_HOOK_PARENT}"

    composed_parent = index.execute(
        "SELECT parent_session_id FROM sessions WHERE session_id = ?", (child_id,)
    ).fetchone()[0]
    assert composed_parent == f"{Origin.CODEX_SESSION.value}:{_HOOK_PARENT}"


def test_revised_hook_claim_supersedes_the_previous_authoritative_edge(tmp_path: Path) -> None:
    """Newest hook claim wins; the old authoritative edge is demoted, not left standing.

    Without this, a revised ``codex_thread_spawn_edge`` lands at a DIFFERENT
    primary key and the previous authoritative edge can be neither purged (it
    is exempt) nor overwritten (the guard refuses), leaving TWO permanent
    authoritative edges and handing composition an arrival-order choice --
    exactly the defect the mechanism exists to remove.
    """
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    _write_spawn_edge(index, parent=_HOOK_PARENT, child=_CHILD)

    write_fixture_index_session(index, _session(_HOOK_PARENT), source_conn=source)
    write_fixture_index_session(index, _session("revised-hook-parent"), source_conn=source)
    child_id = write_fixture_index_session(index, _session(_CHILD), source_conn=source)
    assert _links(index, child_id)[_HOOK_PARENT]["method"] == HOOK_AUTHORITATIVE_LINK_METHOD

    # The projection revises itself: a newer export names a different parent.
    _write_spawn_edge(index, parent="revised-hook-parent", child=_CHILD, observed_at_ms=1_770_000_000_000)

    write_fixture_index_session(index, _session(_CHILD), source_conn=source)

    links = _links(index, child_id)
    authoritative = [name for name, row in links.items() if row["method"] == HOOK_AUTHORITATIVE_LINK_METHOD]
    assert authoritative == ["revised-hook-parent"], "exactly one authoritative parent per child"
    assert (
        index.execute(
            """SELECT COUNT(*) FROM session_links
               WHERE src_session_id = ? AND method = ?""",
            (child_id, HOOK_AUTHORITATIVE_LINK_METHOD),
        ).fetchone()[0]
        == 1
    )
    assert links[_HOOK_PARENT]["method"] == HOOK_SUPERSEDED_LINK_METHOD
    assert links[_HOOK_PARENT]["status"] == TopologyEdgeStatus.AUTHORITY_CONTRADICTED.value
    assert links[_HOOK_PARENT]["resolved_dst_session_id"] is None


def test_unreadable_spawn_edge_projection_is_reported_not_silently_rootless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A projection read that fails must say so before the child lands as a root.

    ``_codex_spawn_edge_parent_claim`` swallowed the failure into
    ``projected_parent = None``, so an unreadable projection archived a Codex
    child with no parent edge and no warning -- indistinguishable from a child
    that genuinely has no parent evidence (polylogue-3r36h). Restoring the
    silent ``except`` turns this red.

    Note the sibling swallow this test does NOT cover:
    ``codex_state_projection.read_parent_thread_id`` catches ``sqlite3.Error``
    at debug level itself, so a broken projection TABLE never reaches the
    handler here. That module is outside this change's scope.
    """
    import sys

    from polylogue.storage.sqlite.archive_tiers.write import _codex_spawn_edge_parent_claim

    conn = _index_conn(tmp_path / "index.db")
    monkeypatch.setitem(sys.modules, "polylogue.sources.codex_state_projection", None)

    with capture() as events:
        claim = _codex_spawn_edge_parent_claim(conn, None, child_native_id="child-thread", child_source_path=None)

    assert claim is None
    assert any(
        event["event"] == "storage.codex_spawn_edge.parent_projection_unavailable"
        and event.get("session_id") == "child-thread"
        for event in events
    )


# ---------------------------------------------------------------------------
# Child archived before the state export that names its parent
# ---------------------------------------------------------------------------


def _project_state_export(
    conn: sqlite3.Connection, *, parent: str, child: str, raw_id: str = "state-raw", observed_at_ms: int = 1_000
) -> None:
    """Land one retained state export through the production projection writer."""
    snapshot = codex_state.CodexStateSnapshot(
        threads=(),
        spawn_edges=(codex_state.CodexSpawnEdge(parent_thread_id=parent, child_thread_id=child, status="closed"),),
    )
    # Callers pass exports in observation order, so the stamp doubles as the
    # durable receipt order that decides which export is current.
    write_thread_state_projection(
        conn,
        snapshot,
        raw_id=raw_id,
        blob_hash=f"blob-{raw_id}",
        observed_at_ms=observed_at_ms,
        observation_order=observed_at_ms,
        source_read=None,
    )
    conn.commit()


def _edge_decisions(conn: sqlite3.Connection, child_id: str) -> dict[str, tuple[object, ...]]:
    return {
        name: (row["link_type"], row["method"], row["status"], row["resolved_dst_session_id"])
        for name, row in _links(conn, child_id).items()
    }


def _composed_parent(conn: sqlite3.Connection, child_id: str) -> tuple[object, object]:
    row = conn.execute(
        "SELECT parent_session_id, session_kind FROM sessions WHERE session_id = ?", (child_id,)
    ).fetchone()
    return row[0], row[1]


@pytest.mark.parametrize("parser_parent", [None, _PARSER_PARENT, _HOOK_PARENT])
def test_state_export_after_the_child_reaches_the_same_topology(tmp_path: Path, parser_parent: str | None) -> None:
    """Topology must not depend on whether the child or its state export landed first.

    Red twin: drop the ``rederive_codex_spawn_parent_links`` call from
    ``write_thread_state_projection`` and the child-first archive keeps the
    parser-only edge (or none) and composes through the wrong parent.
    """
    # Two independent archives, each an Index with its own Source tier.
    state_first = _index_conn(tmp_path / "state-first" / "index.db")
    source = _source_conn(tmp_path / "state-first" / "source.db")
    _project_state_export(state_first, parent=_HOOK_PARENT, child=_CHILD)
    write_fixture_index_session(state_first, _session(_HOOK_PARENT), source_conn=source)
    write_fixture_index_session(state_first, _session(_PARSER_PARENT), source_conn=source)
    expected_child = write_fixture_index_session(
        state_first, _session(_CHILD, parent=parser_parent), source_conn=source
    )

    child_first = _index_conn(tmp_path / "child-first" / "index.db")
    child_source = _source_conn(tmp_path / "child-first" / "source.db")
    write_fixture_index_session(child_first, _session(_HOOK_PARENT), source_conn=child_source)
    write_fixture_index_session(child_first, _session(_PARSER_PARENT), source_conn=child_source)
    child_id = write_fixture_index_session(
        child_first, _session(_CHILD, parent=parser_parent), source_conn=child_source
    )
    before = _edge_decisions(child_first, child_id)
    _project_state_export(child_first, parent=_HOOK_PARENT, child=_CHILD)

    assert child_id == expected_child
    assert _edge_decisions(child_first, child_id) == _edge_decisions(state_first, expected_child) != before
    assert _composed_parent(child_first, child_id) == _composed_parent(state_first, expected_child)
    assert _composed_parent(child_first, child_id)[0] == f"{Origin.CODEX_SESSION.value}:{_HOOK_PARENT}"


def test_revised_state_export_moves_an_already_archived_child(tmp_path: Path) -> None:
    """A newer export naming a different parent re-decides the stored child.

    The parser agreed with the first export, so its edge was upgraded in place;
    the revision must find that parser claim again and mark it contradicted.
    Red twin: drop the ``rederive_codex_spawn_parent_links`` call from
    ``write_thread_state_projection`` and the child keeps composing through
    the first export's parent, because nothing re-saves its transcript.
    """
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    _project_state_export(index, parent=_HOOK_PARENT, child=_CHILD)
    write_fixture_index_session(index, _session(_HOOK_PARENT), source_conn=source)
    write_fixture_index_session(index, _session("revised-hook-parent"), source_conn=source)
    child_id = write_fixture_index_session(index, _session(_CHILD, parent=_HOOK_PARENT), source_conn=source)
    assert _links(index, child_id)[_HOOK_PARENT]["method"] == HOOK_AUTHORITATIVE_LINK_METHOD

    _project_state_export(index, parent="revised-hook-parent", child=_CHILD, raw_id="state-raw-2", observed_at_ms=2_000)

    links = _links(index, child_id)
    assert links["revised-hook-parent"]["method"] == HOOK_AUTHORITATIVE_LINK_METHOD
    assert links["revised-hook-parent"]["status"] is None
    assert links[_HOOK_PARENT]["method"] == HOOK_CONTRADICTED_LINK_METHOD
    assert links[_HOOK_PARENT]["status"] == TopologyEdgeStatus.AUTHORITY_CONTRADICTED.value
    assert links[_HOOK_PARENT]["resolved_dst_session_id"] is None
    assert _composed_parent(index, child_id)[0] == f"{Origin.CODEX_SESSION.value}:revised-hook-parent"


def test_export_returning_to_an_earlier_parent_moves_the_child_back(tmp_path: Path) -> None:
    """A -> B -> A across three exports leaves the child under A.

    Red twin: decide which children to re-derive from the difference of the
    scope's edge-key sets. The graph retains B's superseded edge and A's edge
    already exists, so the third export adds no key and the child keeps B.
    """
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    for parent in (_HOOK_PARENT, "second-hook-parent"):
        write_fixture_index_session(index, _session(parent), source_conn=source)
    child_id = write_fixture_index_session(index, _session(_CHILD), source_conn=source)

    _project_state_export(index, parent=_HOOK_PARENT, child=_CHILD, raw_id="state-a", observed_at_ms=1_000)
    _project_state_export(index, parent="second-hook-parent", child=_CHILD, raw_id="state-b", observed_at_ms=2_000)
    assert _composed_parent(index, child_id)[0] == f"{Origin.CODEX_SESSION.value}:second-hook-parent"
    _project_state_export(index, parent=_HOOK_PARENT, child=_CHILD, raw_id="state-a2", observed_at_ms=3_000)

    links = _links(index, child_id)
    assert links[_HOOK_PARENT]["method"] == HOOK_AUTHORITATIVE_LINK_METHOD
    assert links["second-hook-parent"]["method"] == HOOK_SUPERSEDED_LINK_METHOD
    assert _composed_parent(index, child_id)[0] == f"{Origin.CODEX_SESSION.value}:{_HOOK_PARENT}"


def test_rederivation_uses_the_current_parser_parent_not_a_retired_one(tmp_path: Path) -> None:
    """A parser revision A -> B under hook H, then a hook move H -> C, keeps B as the parser claim.

    Red twin: drop ``_retire_stale_parser_assertions`` from
    ``_write_session_link``. The full replace keeps A's contradicted row, it
    sorts before B's, and the re-derivation recovers A as the parser parent.
    """
    revised_parser_parent = "revised-parser-parent"
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    _project_state_export(index, parent=_HOOK_PARENT, child=_CHILD, raw_id="state-h", observed_at_ms=1_000)
    for parent in (_HOOK_PARENT, _PARSER_PARENT, revised_parser_parent, "moved-hook-parent"):
        write_fixture_index_session(index, _session(parent), source_conn=source)
    write_fixture_index_session(index, _session(_CHILD, parent=_PARSER_PARENT), source_conn=source)
    child_id = write_fixture_index_session(index, _session(_CHILD, parent=revised_parser_parent), source_conn=source)
    links = _links(index, child_id)
    assert links[revised_parser_parent]["method"] == HOOK_CONTRADICTED_LINK_METHOD
    assert _PARSER_PARENT not in links

    _project_state_export(index, parent="moved-hook-parent", child=_CHILD, raw_id="state-c", observed_at_ms=2_000)

    links = _links(index, child_id)
    assert _PARSER_PARENT not in links
    assert links[revised_parser_parent]["method"] == HOOK_CONTRADICTED_LINK_METHOD
    moved = links["moved-hook-parent"]
    assert moved["method"] == HOOK_AUTHORITATIVE_LINK_METHOD
    assert json.loads(moved["evidence_json"])["superseded_parser_parent"] == revised_parser_parent
    assert _composed_parent(index, child_id)[0] == f"{Origin.CODEX_SESSION.value}:moved-hook-parent"


def test_rederiving_a_deep_chain_projects_each_session_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One export that parents a whole chain refreshes the closure in one traversal.

    Red twin: refresh each rewritten child with its own ``seen`` set. Every
    child then climbs its whole ancestor chain again, so the refresh count
    grows with the square of the chain depth instead of linearly.
    """
    from polylogue.storage.sqlite.archive_tiers import write as write_module

    depth = 20
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    chain = ["chain-root", *(f"chain-{position:02d}" for position in range(1, depth + 1))]
    for native_id in chain:
        write_fixture_index_session(index, _session(native_id), source_conn=source)

    calls: list[str] = []
    original = write_module._refresh_session_projection

    def counting(conn: sqlite3.Connection, session_id: str, *, seen: set[str], read: Any = None) -> None:
        calls.append(session_id)
        original(conn, session_id, seen=seen, read=read)

    monkeypatch.setattr(write_module, "_refresh_session_projection", counting)
    snapshot = codex_state.CodexStateSnapshot(
        threads=(),
        spawn_edges=tuple(
            codex_state.CodexSpawnEdge(parent_thread_id=parent, child_thread_id=child, status="closed")
            for parent, child in zip(chain, chain[1:], strict=False)
        ),
    )
    write_thread_state_projection(
        index, snapshot, raw_id="chain", blob_hash="blob-chain", observed_at_ms=1_000, source_read=None
    )
    index.commit()

    assert len(calls) <= 2 * depth
    root_id = f"{Origin.CODEX_SESSION.value}:chain-root"
    rows = index.execute(
        "SELECT session_id, parent_session_id, root_session_id FROM sessions WHERE session_id != ?", (root_id,)
    ).fetchall()
    assert len(rows) == depth
    assert all(row["root_session_id"] == root_id and row["parent_session_id"] is not None for row in rows)


_ROOT_A = "/roots/a/.codex"
_ROOT_B = "/roots/b/.codex"


def _two_roots_naming_one_child(tmp_path: Path) -> tuple[sqlite3.Connection, sqlite3.Connection]:
    """An archive whose child rollout came from root A, with both roots' parents archived."""
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    source.execute(
        "INSERT INTO raw_sessions (raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms) "
        "VALUES ('child-rollout', ?, ?, 0, zeroblob(32), 1, 1)",
        (Origin.CODEX_SESSION.value, f"{_ROOT_A}/sessions/2026/01/01/rollout-{_CHILD}.jsonl"),
    )
    source.commit()
    for parent in ("a-parent", "b-parent"):
        write_fixture_index_session(index, _session(parent), source_conn=source)
    return index, source


def _project_both_roots(index: sqlite3.Connection, source: sqlite3.Connection) -> None:
    """Root A names ``a-parent`` for the child; root B, observed later, names ``b-parent``."""
    for order, (root, parent) in enumerate(((_ROOT_A, "a-parent"), (_ROOT_B, "b-parent")), start=1):
        snapshot = codex_state.CodexStateSnapshot(
            threads=(),
            spawn_edges=(codex_state.CodexSpawnEdge(parent_thread_id=parent, child_thread_id=_CHILD, status="closed"),),
        )
        write_thread_state_projection(
            index,
            snapshot,
            raw_id=f"state-{order}",
            blob_hash=f"blob-{order}",
            observed_at_ms=order * 1_000,
            source_scope=root,
            source_read=ConnectionSessionSourceRead(source),
        )
        index.commit()


def _assert_child_under_root_a(index: sqlite3.Connection, child_id: str) -> None:
    links = _links(index, child_id)
    assert links["a-parent"]["method"] == HOOK_AUTHORITATIVE_LINK_METHOD
    assert links["a-parent"]["status"] is None
    assert "b-parent" not in links
    assert _composed_parent(index, child_id)[0] == f"{Origin.CODEX_SESSION.value}:a-parent"


@pytest.mark.parametrize("state_first", [True, False])
def test_a_child_reads_its_parent_in_its_own_rollouts_install_root(tmp_path: Path, state_first: bool) -> None:
    """Two roots name one child thread with different parents; the save uses its own root's.

    The child's rollout was acquired from root A, so its authoritative parent
    is root A's whether the exports land before the child's save or after it.

    Anti-vacuity: read the projected parent without the child's install
    root and the later root B wins: the save (or the re-derivation after
    root B's export) makes ``b-parent`` authoritative.
    """
    index, source = _two_roots_naming_one_child(tmp_path)
    if state_first:
        _project_both_roots(index, source)
    child_id = write_fixture_index_session(
        index,
        _session(_CHILD),
        source_conn=source,
        raw_id="child-rollout",
        child_source_path=raw_source_path(ConnectionSessionSourceRead(source), "child-rollout"),
    )
    if not state_first:
        _project_both_roots(index, source)

    _assert_child_under_root_a(index, child_id)


def test_a_prepared_child_publishes_under_the_root_it_was_prepared_in(tmp_path: Path) -> None:
    """The install root a prepared write resolved is the root its publication reads.

    The daemon prepares a write against the source tier, while the
    revision-governed write that publishes it holds no source handle; a raw's
    source path never changes, so publication reuses the prepared root.

    Anti-vacuity: re-derive the root at publication instead and the write
    finds none: the two roots disagree, the re-checked claim is silent, and
    the prepared write is refused as stale on every retry.
    """
    from polylogue.pipeline.ids import session_content_hash
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from tests.infra.index_writer import _fixture_writer_admission

    index, source = _two_roots_naming_one_child(tmp_path)
    _project_both_roots(index, source)
    index.commit()
    source.commit()
    child = _session(_CHILD)
    # Preparation resolves the install root through the original Source read;
    # publication runs in that seal's Index scope and reuses the prepared root.
    with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            prepared = prepare_session_write(
                seal.observer("index"),
                child,
                merge_append=False,
                source_read=PreparedSessionSourceRead(seal, blob_store=BlobStore(tmp_path / "blob")),
                raw_id="child-rollout",
                before_input=seal.before_index_input,
            )
        with _fixture_writer_admission(index, "test.prepared-child.publish", tmp_path), seal.mutation_scope(index):
            child_id = write_fixture_index_session(
                index,
                child,
                content_hash=str(session_content_hash(child)),
                prepared_write=prepared,
                raw_id="child-rollout",
            )

    _assert_child_under_root_a(index, child_id)
