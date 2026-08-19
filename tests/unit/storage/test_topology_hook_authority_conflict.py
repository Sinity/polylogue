"""Hook-evidence authority on the LIVE topology write path.

``polylogue-foee`` acquired Codex ``thread_spawn_edges`` into the durable
``source.db`` hook spool as ``codex_thread_spawn_edge`` ``raw_hook_events``,
but nothing consumed them for topology: the only artifact was
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
2. ``session_links`` lives in the REBUILDABLE index tier while hook evidence
   lives in the DURABLE source tier. Only a derivation running inside
   ``write_parsed_session_to_archive`` -- the choke point shared by live
   ingest and full raw replay -- survives a reindex, which is why this is a
   write-path concern rather than a convergence stage.

The losing edge is marked with the EXISTING ``TopologyEdgeStatus.QUARANTINED``
rather than a new status member, and distinguished by its ``method`` token.
That is deliberate: six modules already exclude quarantined edges from
composition with a hardcoded ``!= 'quarantined'`` predicate, so reusing the
member makes exclusion correct by construction, whereas a new member would
require every one of those call sites to be found and updated -- and any miss
would silently readmit a contradicted edge into lineage composition.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.topology.edge import TopologyEdgeStatus
from polylogue.core.enums import BlockType, LinkType, Origin, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    HOOK_AUTHORITATIVE_LINK_METHOD,
    HOOK_CONTRADICTED_LINK_METHOD,
    write_parsed_session_to_archive,
)

_CHILD = "child-thread"
_HOOK_PARENT = "hook-parent-thread"
_PARSER_PARENT = "parser-parent-thread"


def _index_conn(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _source_conn(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.SOURCE)
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


def _write_spawn_edge_event(conn: sqlite3.Connection, *, parent: str, child: str) -> None:
    """Insert one acquired ``codex_thread_spawn_edge`` hook event.

    Mirrors ``sources/codex_state_evidence.py``: keyed by
    ``session_native_id = parent_thread_id``, with both thread ids in the
    payload. The child-side lookup under test must therefore match on the
    payload rather than on ``session_native_id``.
    """
    payload = {"parent_thread_id": parent, "child_thread_id": child, "status": "spawned"}
    conn.execute(
        """
        INSERT INTO raw_hook_events (
            hook_event_id, origin, source_path, event_type, payload_json,
            observed_at_ms, native_id, session_native_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"codex-thread-spawn-edge:{parent}:{child}",
            Origin.CODEX_SESSION.value,
            "/sanitized/state_5.sqlite",
            "codex_thread_spawn_edge",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            1_760_000_000_000,
            f"{parent}:{child}:codex_thread_spawn_edge",
            parent,
        ),
    )
    conn.commit()


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
    _write_spawn_edge_event(source, parent=_HOOK_PARENT, child=_CHILD)

    write_parsed_session_to_archive(index, _session(_HOOK_PARENT), source_conn=source)
    write_parsed_session_to_archive(index, _session(_PARSER_PARENT), source_conn=source)
    child_id = write_parsed_session_to_archive(index, _session(_CHILD, parent=_PARSER_PARENT), source_conn=source)

    links = _links(index, child_id)
    assert set(links) == {_HOOK_PARENT, _PARSER_PARENT}, "both evidence sources must be retained"

    winner = links[_HOOK_PARENT]
    assert winner["status"] is None
    assert winner["method"] == HOOK_AUTHORITATIVE_LINK_METHOD
    assert winner["resolved_dst_session_id"] == f"{Origin.CODEX_SESSION.value}:{_HOOK_PARENT}"

    loser = links[_PARSER_PARENT]
    assert loser["status"] == TopologyEdgeStatus.QUARANTINED.value
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
    _write_spawn_edge_event(source, parent=_HOOK_PARENT, child=_CHILD)

    write_parsed_session_to_archive(index, _session(_HOOK_PARENT), source_conn=source)
    write_parsed_session_to_archive(index, _session(_PARSER_PARENT), source_conn=source)
    child_id = write_parsed_session_to_archive(index, _session(_CHILD, parent=_PARSER_PARENT), source_conn=source)
    assert _links(index, child_id)[_HOOK_PARENT]["method"] == HOOK_AUTHORITATIVE_LINK_METHOD

    # Re-parse with NO hook evidence available at all.
    write_parsed_session_to_archive(index, _session(_CHILD, parent=_PARSER_PARENT), source_conn=None)

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
        (child_id, TopologyEdgeStatus.QUARANTINED.value, HOOK_CONTRADICTED_LINK_METHOD),
    ).fetchone()[0]
    assert quarantined == 1


def test_conflict_state_is_queryable_by_typed_status_and_method(tmp_path: Path) -> None:
    """The durable conflict state is discoverable without parsing prose."""
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    _write_spawn_edge_event(source, parent=_HOOK_PARENT, child=_CHILD)

    write_parsed_session_to_archive(index, _session(_HOOK_PARENT), source_conn=source)
    write_parsed_session_to_archive(index, _session(_PARSER_PARENT), source_conn=source)
    write_parsed_session_to_archive(index, _session(_CHILD, parent=_PARSER_PARENT), source_conn=source)

    conflicts = index.execute(
        """
        SELECT src_session_id, dst_native_id, evidence_json
        FROM session_links
        WHERE status = ? AND method = ?
        """,
        (TopologyEdgeStatus.QUARANTINED.value, HOOK_CONTRADICTED_LINK_METHOD),
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
    _write_spawn_edge_event(source, parent=_HOOK_PARENT, child=_CHILD)

    write_parsed_session_to_archive(index, _session(_HOOK_PARENT), source_conn=source)
    child_id = write_parsed_session_to_archive(index, _session(_CHILD), source_conn=source)

    links = _links(index, child_id)
    assert set(links) == {_HOOK_PARENT}
    assert links[_HOOK_PARENT]["method"] == HOOK_AUTHORITATIVE_LINK_METHOD
    assert links[_HOOK_PARENT]["link_type"] == LinkType.SUBAGENT.value


def test_agreeing_evidence_upgrades_the_single_edge(tmp_path: Path) -> None:
    """Agreement is not a conflict: one edge, marked authoritative."""
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    _write_spawn_edge_event(source, parent=_HOOK_PARENT, child=_CHILD)

    write_parsed_session_to_archive(index, _session(_HOOK_PARENT), source_conn=source)
    child_id = write_parsed_session_to_archive(index, _session(_CHILD, parent=_HOOK_PARENT), source_conn=source)

    links = _links(index, child_id)
    assert set(links) == {_HOOK_PARENT}
    assert links[_HOOK_PARENT]["method"] == HOOK_AUTHORITATIVE_LINK_METHOD
    assert links[_HOOK_PARENT]["status"] is None
    quarantined = index.execute("SELECT COUNT(*) FROM session_links WHERE status IS NOT NULL").fetchone()[0]
    assert quarantined == 0


# ---------------------------------------------------------------------------
# Red twin 3: no hook evidence => byte-identical to today
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_source_handle", [False, True])
def test_no_hook_evidence_is_byte_identical_to_the_parser_only_path(tmp_path: Path, with_source_handle: bool) -> None:
    """Absent evidence is silence, never a conflict.

    Covers both shapes of "no evidence": no source handle at all, and a source
    handle whose spool simply says nothing about this child. Red twin: mark
    edges unconditionally (drop the ``hook_parent is not None`` guards) and
    these rows stop matching the parser-only baseline.
    """
    baseline_index = _index_conn(tmp_path / "baseline.db")
    write_parsed_session_to_archive(baseline_index, _session(_PARSER_PARENT))
    baseline_child = write_parsed_session_to_archive(baseline_index, _session(_CHILD, parent=_PARSER_PARENT))
    baseline = dict(_links(baseline_index, baseline_child)[_PARSER_PARENT])

    index = _index_conn(tmp_path / "index.db")
    source: sqlite3.Connection | None = None
    if with_source_handle:
        source = _source_conn(tmp_path / "source.db")
        # Evidence exists, but about a completely unrelated child.
        _write_spawn_edge_event(source, parent="unrelated-parent", child="unrelated-child")

    write_parsed_session_to_archive(index, _session(_PARSER_PARENT), source_conn=source)
    child_id = write_parsed_session_to_archive(index, _session(_CHILD, parent=_PARSER_PARENT), source_conn=source)
    observed = dict(_links(index, child_id)[_PARSER_PARENT])

    assert observed == baseline
    assert observed["method"] == "parser-parent"
    assert observed["status"] is None
