"""Child session links bind to the exact parent tool_use block that dispatched them.

Evidence enters through the production Claude Code parser: the Agent/Task
tool result record carries ``toolUseResult.agentId`` next to the
``tool_result.tool_use_id`` that names the dispatching block, and the child's
``agent-*.meta.json`` sidecar in the source tier carries ``toolUseId``. The
canonical session-link writer joins those exact keys with the parent's
tool_use block; every refusal is a typed ``dispatch_reason``.

Anti-vacuity for the whole module: restoring an ordinal, nearest-call,
count, or timestamp fallback in ``_resolve_parent_dispatch_block`` makes the
fan-out test bind a child to the wrong block and the refusal tests bind a
block where none may be bound.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.sources.parsers.claude import parse_code
from polylogue.storage.blob_store import get_blob_store
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

_PARENT = "0d9f1c2e-parent-uuid"
_T0 = "2026-05-28T00:59:00.000Z"


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


def _parent_records(
    dispatches: list[tuple[str, str]],
    *,
    with_tool_use: bool = True,
    with_result: bool = True,
    session_id: str = _PARENT,
) -> list[dict[str, object]]:
    """Provider-shaped dispatching transcript: one Agent tool_use + result per dispatch."""
    records: list[dict[str, object]] = [
        {
            "type": "user",
            "uuid": f"{session_id}-u0",
            "sessionId": session_id,
            "timestamp": _T0,
            "message": {"role": "user", "content": "delegate the audit"},
        }
    ]
    for index, (tool_id, agent_id) in enumerate(dispatches):
        if with_tool_use:
            records.append(
                {
                    "type": "assistant",
                    "uuid": f"{session_id}-a{index}",
                    "sessionId": session_id,
                    "timestamp": _T0,
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": "Agent",
                                "input": {"description": f"worker {index}", "prompt": "audit"},
                            }
                        ],
                    },
                }
            )
        if with_result:
            records.append(
                {
                    "type": "user",
                    "uuid": f"{session_id}-r{index}",
                    "sessionId": session_id,
                    "timestamp": _T0,
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "tool_use_id": tool_id,
                                "type": "tool_result",
                                "content": [{"type": "text", "text": "Async agent launched successfully."}],
                            }
                        ],
                    },
                    "toolUseResult": {
                        "isAsync": True,
                        "status": "async_launched",
                        "agentId": agent_id,
                        "description": f"worker {index}",
                    },
                }
            )
    return records


def _child_records(agent_id: str, *, parent: str = _PARENT) -> list[dict[str, object]]:
    return [
        {
            "parentUuid": None,
            "isSidechain": True,
            "promptId": "prompt-1",
            "agentId": agent_id,
            "type": "user",
            "uuid": f"{agent_id}-u0",
            "sessionId": parent,
            "timestamp": _T0,
            "message": {"role": "user", "content": f"You are worker {agent_id}."},
        },
        {
            "parentUuid": f"{agent_id}-u0",
            "isSidechain": True,
            "agentId": agent_id,
            "type": "assistant",
            "uuid": f"{agent_id}-a0",
            "sessionId": parent,
            "timestamp": _T0,
            "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        },
    ]


def _write_parent(conn: sqlite3.Connection, records: list[dict[str, object]], **kwargs: object) -> str:
    return write_parsed_session_to_archive(conn, parse_code(records, _PARENT), **kwargs)  # type: ignore[arg-type]


def _write_child(conn: sqlite3.Connection, agent_id: str, **kwargs: object) -> str:
    return write_parsed_session_to_archive(conn, parse_code(_child_records(agent_id), f"agent-{agent_id}"), **kwargs)  # type: ignore[arg-type]


def _link(conn: sqlite3.Connection, child_id: str) -> sqlite3.Row:
    rows = conn.execute(
        """SELECT resolved_dst_session_id, parent_tool_use_block_id, method, evidence_json
           FROM session_links WHERE src_session_id = ?""",
        (child_id,),
    ).fetchall()
    assert len(rows) == 1, rows
    return rows[0]


def _tool_use_block_id(conn: sqlite3.Connection, tool_id: str) -> str:
    rows = conn.execute(
        "SELECT block_id FROM blocks WHERE tool_id = ? AND block_type = 'tool_use'", (tool_id,)
    ).fetchall()
    assert len(rows) == 1, rows
    return str(rows[0][0])


def _dispatch_reason(row: sqlite3.Row) -> str | None:
    value = json.loads(row["evidence_json"]).get("dispatch_reason")
    return None if value is None else str(value)


def test_parent_first_binds_child_to_exact_dispatch_block(tmp_path: Path) -> None:
    """Red if ``toolUseResult.agentId`` stops lowering into the dispatch observation."""
    conn = _index_conn(tmp_path / "index.db")
    parent_id = _write_parent(conn, _parent_records([("call_1", "a1")]))
    child_id = _write_child(conn, "a1")

    link = _link(conn, child_id)
    assert link["resolved_dst_session_id"] == parent_id
    assert link["parent_tool_use_block_id"] == _tool_use_block_id(conn, "call_1")
    assert link["method"] == "parent-tool-use-id"
    assert _dispatch_reason(link) is None


def test_child_first_converges_to_the_same_edge(tmp_path: Path) -> None:
    """Order independence: the child arriving before its parent yields an identical edge.

    Red if ``_refill_inbound_dispatch_block_ids`` stops running on the parent
    write, or if the inbound resolution loop stops binding the block.
    """
    first = _index_conn(tmp_path / "parent-first.db")
    _write_parent(first, _parent_records([("call_1", "a1")]))
    child_first_edge = dict(_link(first, _write_child(first, "a1")))

    second = _index_conn(tmp_path / "child-first.db")
    child_id = _write_child(second, "a1")
    pending = _link(second, child_id)
    assert pending["resolved_dst_session_id"] is None
    assert pending["parent_tool_use_block_id"] is None
    _write_parent(second, _parent_records([("call_1", "a1")]))
    parent_first_edge = dict(_link(second, child_id))

    assert parent_first_edge == child_first_edge
    assert parent_first_edge["parent_tool_use_block_id"] == _tool_use_block_id(second, "call_1")


def test_fan_out_binds_each_child_to_its_own_block(tmp_path: Path) -> None:
    """Two dispatches in one parent stay distinguishable by provider tool id.

    Red under any ordinal or nearest-call pairing: the children are written
    in reverse dispatch order, so a positional guess swaps the blocks.
    """
    conn = _index_conn(tmp_path / "index.db")
    _write_parent(conn, _parent_records([("call_1", "a1"), ("call_2", "a2")]))
    second_child = _write_child(conn, "a2")
    first_child = _write_child(conn, "a1")

    assert _link(conn, first_child)["parent_tool_use_block_id"] == _tool_use_block_id(conn, "call_1")
    assert _link(conn, second_child)["parent_tool_use_block_id"] == _tool_use_block_id(conn, "call_2")


def test_missing_dispatch_evidence_is_typed_absent(tmp_path: Path) -> None:
    """A parent whose result record lacks the child identity refuses, and says why.

    Red if the resolver falls back to the parent's only Agent tool_use block.
    """
    conn = _index_conn(tmp_path / "index.db")
    _write_parent(conn, _parent_records([("call_1", "a1")], with_result=False))
    child_id = _write_child(conn, "a1")

    link = _link(conn, child_id)
    assert link["resolved_dst_session_id"] is not None
    assert link["parent_tool_use_block_id"] is None
    assert link["method"] == "parser-parent"
    assert _dispatch_reason(link) == "dispatch-evidence-absent"


def test_evidence_naming_an_absent_block_is_typed_missing(tmp_path: Path) -> None:
    """Red if a named-but-absent block degrades to silent NULL."""
    conn = _index_conn(tmp_path / "index.db")
    _write_parent(conn, _parent_records([("call_1", "a1")], with_tool_use=False))
    child_id = _write_child(conn, "a1")

    link = _link(conn, child_id)
    assert link["parent_tool_use_block_id"] is None
    assert _dispatch_reason(link) == "dispatch-block-missing"


def test_conflicting_witnesses_are_contradicted_not_guessed(tmp_path: Path) -> None:
    """Two tool ids naming the same child never resolve to either block."""
    conn = _index_conn(tmp_path / "index.db")
    _write_parent(conn, _parent_records([("call_1", "a1"), ("call_2", "a1")]))
    child_id = _write_child(conn, "a1")

    link = _link(conn, child_id)
    assert link["parent_tool_use_block_id"] is None
    assert _dispatch_reason(link) == "dispatch-identity-contradiction"


def _seed_sidecar(source: sqlite3.Connection, *, parent_dir: str, agent_id: str, tool_use_id: str) -> None:
    payload = json.dumps(
        {"agentType": "general-purpose", "description": "worker", "toolUseId": tool_use_id, "spawnDepth": 1}
    ).encode()
    hash_hex, size = get_blob_store().write_from_bytes(payload)
    source.execute(
        """INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms)
           VALUES (?, 'claude-code-session', ?, ?, ?, 0)""",
        (
            f"raw-{parent_dir}-{agent_id}",
            f"/x/.claude/projects/proj/{parent_dir}/subagents/agent-{agent_id}.meta.json",
            bytes.fromhex(hash_hex),
            size,
        ),
    )
    source.commit()


def test_sidecar_tool_use_id_binds_when_the_parent_result_is_silent(tmp_path: Path) -> None:
    """The child's ``agent-*.meta.json`` sidecar is an exact witness from the source tier.

    A decoy sidecar for the same child stem under a different parent
    directory names another tool id; it must not bind, or contradict, this
    edge. Red if ``parse_claude_orchestration_artifact`` drops ``toolUseId``
    or if the parent-directory binding is removed (the decoy would then
    contradict the real witness).
    """
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    _seed_sidecar(source, parent_dir=_PARENT, agent_id="a1", tool_use_id="call_1")
    _seed_sidecar(source, parent_dir="some-other-parent", agent_id="a1", tool_use_id="call_9")

    _write_parent(index, _parent_records([("call_1", "a1")], with_result=False), source_conn=source)
    child_id = _write_child(index, "a1", source_conn=source)

    link = _link(index, child_id)
    assert link["parent_tool_use_block_id"] == _tool_use_block_id(index, "call_1")
    assert link["method"] == "parent-tool-use-id"
    assert _dispatch_reason(link) is None


def test_sidecar_and_parent_result_disagreeing_is_a_contradiction(tmp_path: Path) -> None:
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    _seed_sidecar(source, parent_dir=_PARENT, agent_id="a1", tool_use_id="call_2")

    _write_parent(index, _parent_records([("call_1", "a1"), ("call_2", "a2")]), source_conn=source)
    child_id = _write_child(index, "a1", source_conn=source)

    assert _link(index, child_id)["parent_tool_use_block_id"] is None
    assert _dispatch_reason(_link(index, child_id)) == "dispatch-identity-contradiction"


def test_delegation_facts_consume_the_canonical_edge(tmp_path: Path) -> None:
    """The delegation projection reads the bound block; it does not pair on its own.

    Red if ``delegation_facts_source`` regains any join other than
    ``parent_tool_use_block_id = instruction_tool_use_block_id``: the
    unresolved second dispatch would then be paired with the only child.
    """
    conn = _index_conn(tmp_path / "index.db")
    parent_id = _write_parent(conn, _parent_records([("call_1", "a1"), ("call_2", "a2")]))
    child_id = _write_child(conn, "a1")

    rows = conn.execute(
        """SELECT mapping_state, child_session_id, instruction_tool_use_block_id
           FROM delegation_facts WHERE parent_session_id = ?
           ORDER BY instruction_tool_use_block_id""",
        (parent_id,),
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("resolved", child_id, _tool_use_block_id(conn, "call_1")),
        ("unresolved", None, _tool_use_block_id(conn, "call_2")),
    ]


def test_origin_without_dispatch_identity_is_typed(tmp_path: Path) -> None:
    """Codex declares no parent-dispatch identity; the refusal names the origin, not the evidence."""
    conn = _index_conn(tmp_path / "index.db")

    def _session(native_id: str, *, parent: str | None = None) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id,
            parent_session_provider_id=parent,
            messages=[
                ParsedMessage(
                    provider_message_id=f"{native_id}-m0",
                    role=Role.USER,
                    text="work",
                    position=0,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="work")],
                )
            ],
        )

    write_parsed_session_to_archive(conn, _session("codex-parent"))
    child_id = write_parsed_session_to_archive(conn, _session("codex-child", parent="codex-parent"))

    link = _link(conn, child_id)
    assert link["resolved_dst_session_id"] is not None
    assert link["parent_tool_use_block_id"] is None
    assert _dispatch_reason(link) == "origin-no-dispatch-identity"
