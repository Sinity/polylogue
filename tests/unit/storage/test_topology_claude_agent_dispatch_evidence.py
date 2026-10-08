"""Claude Code hook ``agent_id``/``agent_type`` as written subagent lineage.

Claude Code stamps ``agent_id`` (the agent instance) and ``agent_type`` (the
agent definition) on the ``PreToolUse``/``PostToolUse`` payloads of the calls a
dispatched agent itself made, in the DISPATCHING session's hook journal. The
dispatching ``Agent`` call carries no such pair, so the pair identifies the
child rather than the dispatch block.

The evidence lands on the write path, not in a convergence stage:
``session_links`` is rebuildable while the hook spool is durable, so only a
derivation inside ``write_parsed_session_to_archive`` survives a reindex.

Anti-vacuity for the whole file: strip ``agent_id``/``agent_type`` from the
payload (or point the hook's ``tool_use_id`` at a call this child never made)
and the edge falls back to the parser method with no hook evidence recorded --
``test_pair_absent_leaves_the_parser_edge_unqualified`` and
``test_tool_use_id_must_name_one_of_this_child_s_own_calls`` are those twins.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.topology.edge import (
    HOOK_AUTHORITATIVE_LINK_METHOD,
    HOOK_CONTRADICTED_LINK_METHOD,
    TopologyEdgeStatus,
)
from polylogue.core.enums import BlockType, Origin, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.write import ConnectionSessionSourceRead
from tests.infra.archive_templates import bootstrapped_tier_path
from tests.infra.index_writer import write_fixture_index_session, write_fixture_prepared_session

_PARENT = "8f6c4d02-1f4a-4f2f-9a1e-1b2c3d4e5f60"
_OTHER_PARENT = "3b7e9a15-6c2d-4e8f-8a0b-7d6c5b4a3f21"
_AGENT_ID = "alog-consolidator-d4b9429a916062bb"
_AGENT_TYPE = "log-consolidator"
_CHILD_STEM = f"agent-{_AGENT_ID}"
_CHILD = f"{_PARENT}:{_CHILD_STEM}"
_TOOL_USE_ID = "toolu_017BvcpTBFAstUygpPFiMaX6"
_OBSERVED_AT_MS = 1_760_000_000_000


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


def _parent_session(native_id: str = _PARENT) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=native_id,
        title=native_id,
        messages=[
            ParsedMessage(
                provider_message_id=f"{native_id}-0",
                role=Role.USER,
                text="dispatch a log consolidator",
                position=0,
                variant_index=0,
                is_active_path=True,
                is_active_leaf=False,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="dispatch a log consolidator")],
            )
        ],
    )


def _child_session(*, tool_use_id: str = _TOOL_USE_ID, parent: str = _PARENT) -> ParsedSession:
    """The dispatched child, carrying the tool call the hook attributes to it."""
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=_CHILD,
        title=_CHILD_STEM,
        parent_session_provider_id=parent,
        provider_session_aliases=[_CHILD_STEM],
        branch_type=BranchType.SUBAGENT,
        messages=[
            ParsedMessage(
                provider_message_id=f"{_CHILD}-0",
                role=Role.ASSISTANT,
                text="running the consolidation",
                position=0,
                variant_index=0,
                is_active_path=True,
                is_active_leaf=False,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Bash",
                        tool_id=tool_use_id,
                        tool_input={"command": "true"},
                    )
                ],
            )
        ],
    )


def _write_tool_hook_event(
    conn: sqlite3.Connection,
    *,
    payload: dict[str, object],
    event_type: str = "PostToolUse",
    event_id: str = "e1",
) -> None:
    """Insert one materialized hook envelope exactly as ``sources/hooks`` stores it.

    ``carrier_hook_events`` builds the row from the producer's own carrier
    line, so the harness payload sits under ``$.payload`` and the reader must
    resolve it there.
    """
    record = {
        "event_type": event_type,
        "session_id": _PARENT,
        "provider": "claude-code",
        "timestamp": "2026-09-06T12:00:00Z",
        "event_id": event_id,
        "payload": payload,
    }
    conn.execute(
        """
        INSERT INTO raw_hook_events (
            hook_event_id, origin, source_path, event_type, payload_json,
            observed_at_ms, native_id, session_native_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"hook:{event_id}",
            Origin.CLAUDE_CODE_SESSION.value,
            f"/sanitized/hooks/carriers/claude-code/2026-09-06/{event_id}.ndjson",
            event_type,
            json.dumps(record, sort_keys=True, separators=(",", ":")),
            _OBSERVED_AT_MS,
            f"{_PARENT}:{event_type}:{event_id}",
            _PARENT,
        ),
    )
    conn.commit()


def _snake_payload(*, tool_use_id: str = _TOOL_USE_ID) -> dict[str, object]:
    return {
        "hook_event_name": "PostToolUse",
        "session_id": _PARENT,
        "agent_id": _AGENT_ID,
        "agent_type": _AGENT_TYPE,
        "tool_name": "Bash",
        "tool_use_id": tool_use_id,
    }


def _edge(conn: sqlite3.Connection, src_session_id: str) -> sqlite3.Row:
    rows: list[sqlite3.Row] = conn.execute(
        """
        SELECT dst_native_id, link_type, status, method, resolved_dst_session_id, evidence_json
        FROM session_links WHERE src_session_id = ?
        """,
        (src_session_id,),
    ).fetchall()
    assert len(rows) == 1, f"expected one parent edge, got {[dict(row) for row in rows]}"
    return rows[0]


def _ingest(
    tmp_path: Path,
    payloads: list[dict[str, object]],
    *,
    child_tool_use_id: str = _TOOL_USE_ID,
) -> tuple[sqlite3.Connection, str]:
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    for position, payload in enumerate(payloads):
        _write_tool_hook_event(source, payload=payload, event_id=f"e{position}")
    write_fixture_index_session(index, _parent_session(), source_conn=source)
    child = _child_session(tool_use_id=child_tool_use_id)
    child_id = write_fixture_index_session(index, child, source_conn=source)
    return index, child_id


def test_hook_pair_marks_the_subagent_edge_authoritative(tmp_path: Path) -> None:
    """The runtime's own attribution decides the edge, and stays on the row."""
    index, child_id = _ingest(tmp_path, [_snake_payload()])

    edge = _edge(index, child_id)
    assert edge["dst_native_id"] == _PARENT
    assert edge["method"] == HOOK_AUTHORITATIVE_LINK_METHOD
    assert edge["status"] is None
    assert edge["resolved_dst_session_id"] == f"{Origin.CLAUDE_CODE_SESSION.value}:{_PARENT}"

    evidence = json.loads(edge["evidence_json"])
    assert evidence["claude_hook_agent_id"] == _AGENT_ID
    assert evidence["claude_hook_agent_type"] == _AGENT_TYPE
    assert evidence["claude_hook_tool_use_id_matches"] == 1


def test_camel_case_generation_reads_the_same_pair(tmp_path: Path) -> None:
    """The harness's camelCase generation is the same evidence, not an absence."""
    payload: dict[str, object] = {
        "hookEventName": "PostToolUse",
        "sessionId": _PARENT,
        "agentId": _AGENT_ID,
        "agentType": _AGENT_TYPE,
        "toolName": "Bash",
        "toolUseId": _TOOL_USE_ID,
    }
    index, child_id = _ingest(tmp_path, [payload])

    edge = _edge(index, child_id)
    assert edge["method"] == HOOK_AUTHORITATIVE_LINK_METHOD
    assert json.loads(edge["evidence_json"])["claude_hook_agent_type"] == _AGENT_TYPE


def test_pair_absent_leaves_the_parser_edge_unqualified(tmp_path: Path) -> None:
    """A hook event for the same call without the pair asserts nothing.

    This is the anti-vacuity twin: the event, the session and the
    ``tool_use_id`` are all present, so only the two keys under test can be
    responsible for the authoritative marking.
    """
    payload: dict[str, object] = {
        "hook_event_name": "PostToolUse",
        "session_id": _PARENT,
        "tool_name": "Bash",
        "tool_use_id": _TOOL_USE_ID,
    }
    index, child_id = _ingest(tmp_path, [payload])

    edge = _edge(index, child_id)
    assert edge["method"] != HOOK_AUTHORITATIVE_LINK_METHOD
    assert "claude_hook_agent_id" not in json.loads(edge["evidence_json"])


def test_tool_use_id_must_name_one_of_this_child_s_own_calls(tmp_path: Path) -> None:
    """The pair binds through ``tool_use_id``, never through the name alone."""
    index, child_id = _ingest(tmp_path, [_snake_payload(tool_use_id="toolu_someone_else")])

    edge = _edge(index, child_id)
    assert edge["method"] != HOOK_AUTHORITATIVE_LINK_METHOD
    assert "claude_hook_agent_id" not in json.loads(edge["evidence_json"])


def test_hook_evidence_names_the_agent_instance_that_ran_the_call(tmp_path: Path) -> None:
    """A second instance's calls in the same journal do not claim this child."""
    other: dict[str, object] = dict(_snake_payload(tool_use_id="toolu_other_instance"))
    other["agent_id"] = "aexplore-0000000000000000"
    other["agent_type"] = "Explore"
    index, child_id = _ingest(tmp_path, [other, _snake_payload()])

    evidence = json.loads(_edge(index, child_id)["evidence_json"])
    assert evidence["claude_hook_agent_id"] == _AGENT_ID
    assert evidence["claude_hook_agent_type"] == _AGENT_TYPE


def test_reparse_without_the_source_tier_cannot_downgrade_the_edge(tmp_path: Path) -> None:
    """The marking survives an index-only reprocess, hence a reindex."""
    index, child_id = _ingest(tmp_path, [_snake_payload()])
    assert _edge(index, child_id)["method"] == HOOK_AUTHORITATIVE_LINK_METHOD

    write_fixture_index_session(index, _child_session(), source_conn=None)

    edge = _edge(index, child_id)
    assert edge["method"] == HOOK_AUTHORITATIVE_LINK_METHOD
    assert json.loads(edge["evidence_json"])["claude_hook_agent_type"] == _AGENT_TYPE


def test_parser_parent_move_keeps_the_preserved_hook_parent(tmp_path: Path) -> None:
    """A durable hook claim under A still decides the edge after the parser names B.

    The spool is keyed by the dispatching session, so a claim is confirmed
    only under a parent someone names. Red twin: consult only the parser's
    current candidate. B's journal is silent, B's edge lands as an ordinary
    status-NULL parser edge beside the preserved authoritative A edge, and the
    child has two composing parents.
    """
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    _write_tool_hook_event(source, payload=_snake_payload())
    dispatch_tool_id = "toolu_parent_dispatch"
    parent = _parent_session()
    dispatching_parent = parent.model_copy(
        update={
            "messages": [
                *parent.messages,
                ParsedMessage(
                    provider_message_id=f"{_PARENT}-1",
                    role=Role.ASSISTANT,
                    text="dispatching",
                    position=1,
                    variant_index=0,
                    is_active_path=True,
                    is_active_leaf=False,
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TOOL_USE,
                            tool_name="Agent",
                            tool_id=dispatch_tool_id,
                            tool_input={"prompt": "consolidate the logs"},
                        )
                    ],
                ),
            ],
            "session_events": [
                ParsedSessionEvent(
                    event_type="claude_delegation_progress",
                    source_message_provider_id=dispatch_tool_id,
                    payload={"child_provider_id": _CHILD},
                )
            ],
        }
    )
    write_fixture_index_session(index, dispatching_parent, source_conn=source)
    write_fixture_index_session(index, _parent_session(_OTHER_PARENT), source_conn=source)
    child_id = write_fixture_index_session(index, _child_session(), source_conn=source)
    assert _edge(index, child_id)["method"] == HOOK_AUTHORITATIVE_LINK_METHOD
    dispatch_block_id = index.execute(
        "SELECT block_id FROM blocks WHERE tool_id = ? AND block_type = 'tool_use'", (dispatch_tool_id,)
    ).fetchone()[0]
    bound = index.execute(
        "SELECT parent_tool_use_block_id FROM session_links WHERE src_session_id = ?", (child_id,)
    ).fetchone()[0]
    assert bound == dispatch_block_id

    write_fixture_index_session(index, _child_session(parent=_OTHER_PARENT), source_conn=source)

    links = {
        str(row["dst_native_id"]): row
        for row in index.execute(
            """
            SELECT dst_native_id, status, method, resolved_dst_session_id, parent_tool_use_block_id, evidence_json
            FROM session_links WHERE src_session_id = ?
            """,
            (child_id,),
        ).fetchall()
    }
    assert set(links) == {_PARENT, _OTHER_PARENT}
    hook_edge = links[_PARENT]
    assert hook_edge["method"] == HOOK_AUTHORITATIVE_LINK_METHOD
    assert hook_edge["status"] is None
    # The block was resolved against A; B, the contradicted parser parent, has
    # no dispatch evidence and must not unbind it.
    assert hook_edge["parent_tool_use_block_id"] == dispatch_block_id
    assert hook_edge["resolved_dst_session_id"] == f"{Origin.CLAUDE_CODE_SESSION.value}:{_PARENT}"
    hook_evidence = json.loads(hook_edge["evidence_json"])
    assert hook_evidence["claude_hook_agent_id"] == _AGENT_ID
    assert hook_evidence["superseded_parser_parent"] == _OTHER_PARENT

    parser_edge = links[_OTHER_PARENT]
    assert parser_edge["method"] == HOOK_CONTRADICTED_LINK_METHOD
    assert parser_edge["status"] == TopologyEdgeStatus.AUTHORITY_CONTRADICTED.value
    assert parser_edge["resolved_dst_session_id"] is None

    composed_parent = index.execute(
        "SELECT parent_session_id FROM sessions WHERE session_id = ?", (child_id,)
    ).fetchone()[0]
    assert composed_parent == f"{Origin.CLAUDE_CODE_SESSION.value}:{_PARENT}"


def test_parser_parent_with_its_own_hook_claim_supersedes_the_preserved_one(tmp_path: Path) -> None:
    """The candidate is asked first: when B's journal claims the agent, B wins.

    This pins the order the preserved-edge lookup must not invert. B's claim
    agrees with the parser, so the A edge is superseded rather than kept as a
    competing authority.
    """
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    _write_tool_hook_event(source, payload=_snake_payload())
    record = {
        "event_type": "PostToolUse",
        "session_id": _OTHER_PARENT,
        "provider": "claude-code",
        "timestamp": "2026-09-06T12:05:00Z",
        "event_id": "e-other",
        "payload": {**_snake_payload(), "session_id": _OTHER_PARENT},
    }
    source.execute(
        """
        INSERT INTO raw_hook_events (
            hook_event_id, origin, source_path, event_type, payload_json,
            observed_at_ms, native_id, session_native_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "hook:e-other",
            Origin.CLAUDE_CODE_SESSION.value,
            "/sanitized/hooks/carriers/claude-code/2026-09-06/e-other.ndjson",
            "PostToolUse",
            json.dumps(record, sort_keys=True, separators=(",", ":")),
            _OBSERVED_AT_MS + 1,
            f"{_OTHER_PARENT}:PostToolUse:e-other",
            _OTHER_PARENT,
        ),
    )
    source.commit()
    write_fixture_index_session(index, _parent_session(), source_conn=source)
    write_fixture_index_session(index, _parent_session(_OTHER_PARENT), source_conn=source)
    child_id = write_fixture_index_session(index, _child_session(), source_conn=source)
    write_fixture_index_session(index, _child_session(parent=_OTHER_PARENT), source_conn=source)

    authoritative = [
        str(row[0])
        for row in index.execute(
            "SELECT dst_native_id FROM session_links WHERE src_session_id = ? AND method = ? AND status IS NULL",
            (child_id, HOOK_AUTHORITATIVE_LINK_METHOD),
        ).fetchall()
    ]
    assert authoritative == [_OTHER_PARENT]
    composed_parent = index.execute(
        "SELECT parent_session_id FROM sessions WHERE session_id = ?", (child_id,)
    ).fetchone()[0]
    assert composed_parent == f"{Origin.CLAUDE_CODE_SESSION.value}:{_OTHER_PARENT}"


@pytest.mark.parametrize("source_available", [False, True])
@pytest.mark.parametrize("route", ["inline", "prepared"])
def test_preserved_hook_parent_controls_prefix_slicing(tmp_path: Path, source_available: bool, route: str) -> None:
    from contextlib import closing

    from polylogue.storage.sqlite.archive_tiers.write import prepared_lineage_bindings

    with closing(_index_conn(tmp_path / "index.db")) as index, closing(_source_conn(tmp_path / "source.db")) as source:
        _write_tool_hook_event(source, payload=_snake_payload())
        child = _child_session()
        prefix = ParsedMessage(
            provider_message_id="child-prefix",
            role=Role.USER,
            text="only child and B share this",
            position=0,
            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="only child and B share this")],
        )
        child = child.model_copy(update={"messages": [prefix, child.messages[0].model_copy(update={"position": 1})]})
        parent_b = _parent_session(_OTHER_PARENT).model_copy(update={"messages": [prefix]})
        write_fixture_index_session(index, _parent_session(), source_conn=source)
        write_fixture_index_session(index, parent_b, source_conn=source)
        child_id = write_fixture_index_session(index, child, source_conn=source)
        assert (
            index.execute("SELECT method FROM session_links WHERE src_session_id = ?", (child_id,)).fetchone()[0]
            == HOOK_AUTHORITATIVE_LINK_METHOD
        )
        replay = child.model_copy(update={"parent_session_provider_id": _OTHER_PARENT})
        replay_source = source if source_available else None
        assert prepared_lineage_bindings(
            index, replay, source_read=None if replay_source is None else ConnectionSessionSourceRead(replay_source)
        ) == (
            _PARENT,
            f"{Origin.CLAUDE_CODE_SESSION.value}:{_PARENT}",
        )
        if route == "prepared":
            # Canonical preparation always reads its seal's original Source
            # snapshot; source availability varies only the inline route and
            # the lineage-binding read above.
            write_fixture_prepared_session(index, replay)
        else:
            write_fixture_index_session(index, replay, source_conn=replay_source)
        rows = index.execute(
            "SELECT dst_native_id, inheritance, branch_point_message_id FROM session_links "
            "WHERE src_session_id = ? AND status IS NULL",
            (child_id,),
        ).fetchall()
        assert [tuple(row) for row in rows] == [(_PARENT, "spawned-fresh", None)]
        assert index.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (child_id,)).fetchone()[0] == 2
        assert (
            index.execute("SELECT parent_session_id FROM sessions WHERE session_id = ?", (child_id,)).fetchone()[0]
            == f"{Origin.CLAUDE_CODE_SESSION.value}:{_PARENT}"
        )
