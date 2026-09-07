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

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.topology.edge import HOOK_AUTHORITATIVE_LINK_METHOD
from polylogue.core.enums import BlockType, Origin, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

_PARENT = "8f6c4d02-1f4a-4f2f-9a1e-1b2c3d4e5f60"
_AGENT_ID = "alog-consolidator-d4b9429a916062bb"
_AGENT_TYPE = "log-consolidator"
_CHILD_STEM = f"agent-{_AGENT_ID}"
_CHILD = f"{_PARENT}:{_CHILD_STEM}"
_TOOL_USE_ID = "toolu_017BvcpTBFAstUygpPFiMaX6"
_OBSERVED_AT_MS = 1_760_000_000_000


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


def _parent_session() -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=_PARENT,
        title=_PARENT,
        messages=[
            ParsedMessage(
                provider_message_id=f"{_PARENT}-0",
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


def _child_session(*, tool_use_id: str = _TOOL_USE_ID) -> ParsedSession:
    """The dispatched child, carrying the tool call the hook attributes to it."""
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=_CHILD,
        title=_CHILD_STEM,
        parent_session_provider_id=_PARENT,
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
    """Insert one drained hook envelope exactly as ``sources/hooks`` stores it.

    ``_persist_record`` writes the producer envelope, so the harness payload
    sits under ``$.payload`` and the reader must resolve it there.
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
            f"/sanitized/hooks/pending/{event_id}.json",
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
    write_parsed_session_to_archive(index, _parent_session(), source_conn=source)
    child = _child_session(tool_use_id=child_tool_use_id)
    child_id = write_parsed_session_to_archive(index, child, source_conn=source)
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

    write_parsed_session_to_archive(index, _child_session(), source_conn=None)

    edge = _edge(index, child_id)
    assert edge["method"] == HOOK_AUTHORITATIVE_LINK_METHOD
    assert json.loads(edge["evidence_json"])["claude_hook_agent_type"] == _AGENT_TYPE
