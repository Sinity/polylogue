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

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import cast

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.sources.parsers.claude import parse_code
from polylogue.storage.blob_store import blob_store_for_connection
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.index_writer import write_fixture_index_session
from tests.infra.retained_replay import replay_retained_components

_PARENT = "0d9f1c2e-parent-uuid"
_T0 = "2026-05-28T00:59:00.000Z"


def _bootstrapped(path: Path) -> Path:
    # Tier files live inside a canonical archive root; the fixture writer
    # refuses tier files that predate the root's format marker.
    if not (path.parent / ".polylogue-format.json").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap_archive_root(path.parent)
    return path


def _index_conn(path: Path) -> sqlite3.Connection:
    conn = connect_measured(_bootstrapped(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _source_conn(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(_bootstrapped(path))
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
    return write_fixture_index_session(conn, parse_code(records, _PARENT), **kwargs)  # type: ignore[arg-type]


def _write_child(conn: sqlite3.Connection, agent_id: str, **kwargs: object) -> str:
    return write_fixture_index_session(conn, parse_code(_child_records(agent_id), f"agent-{agent_id}"), **kwargs)  # type: ignore[arg-type]


def _link(conn: sqlite3.Connection, child_id: str) -> sqlite3.Row:
    rows = conn.execute(
        """SELECT resolved_dst_session_id, parent_tool_use_block_id, method, evidence_json
           FROM session_links WHERE src_session_id = ?""",
        (child_id,),
    ).fetchall()
    assert len(rows) == 1, rows
    return cast(sqlite3.Row, rows[0])


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
    first = _index_conn(tmp_path / "parent-first" / "index.db")
    _write_parent(first, _parent_records([("call_1", "a1")]))
    child_first_edge = dict(_link(first, _write_child(first, "a1")))

    second = _index_conn(tmp_path / "child-first" / "index.db")
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


def _sidecar_payload(tool_use_id: str) -> bytes:
    return json.dumps(
        {"agentType": "general-purpose", "description": "worker", "toolUseId": tool_use_id, "spawnDepth": 1}
    ).encode()


def _seed_raw(source: sqlite3.Connection, *, raw_id: str, source_path: str, payload: bytes) -> str:
    """One Claude Code raw acquisition, its bytes in this archive's own CAS."""
    hash_hex, size = blob_store_for_connection(source).write_from_bytes(payload)
    source.execute(
        """INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms)
           VALUES (?, 'claude-code-session', ?, ?, ?, 0)""",
        (raw_id, source_path, bytes.fromhex(hash_hex), size),
    )
    source.commit()
    return raw_id


def _subagent_path(parent_dir: str, agent_id: str, suffix: str) -> str:
    return f"/x/.claude/projects/proj/{parent_dir}/subagents/agent-{agent_id}{suffix}"


def _seed_sidecar(source: sqlite3.Connection, *, parent_dir: str, agent_id: str, tool_use_id: str) -> None:
    _seed_raw(
        source,
        raw_id=f"raw-{parent_dir}-{agent_id}",
        source_path=_subagent_path(parent_dir, agent_id, ".meta.json"),
        payload=_sidecar_payload(tool_use_id),
    )


def _seed_child_transcript(source: sqlite3.Connection, agent_id: str, *, parent_dir: str = _PARENT) -> str:
    """The child's own transcript acquisition; its sidecar is this file's sibling."""
    return _seed_raw(
        source,
        raw_id=f"raw-transcript-{parent_dir}-{agent_id}",
        source_path=_subagent_path(parent_dir, agent_id, ".jsonl"),
        payload=b"\n".join(json.dumps(record).encode() for record in _child_records(agent_id)),
    )


def test_sidecar_tool_use_id_binds_when_the_parent_result_is_silent(tmp_path: Path) -> None:
    """The child's ``agent-*.meta.json`` sidecar is an exact witness from the source tier.

    A decoy sidecar for the same child stem under a different parent
    directory names another tool id; it must not bind, or contradict, this
    edge. Red if ``parse_claude_orchestration_artifact`` drops ``toolUseId``
    or if the sidecar is looked up by child stem alone rather than at the
    path the child's own acquisition names (the decoy would then contradict
    the real witness).
    """
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    _seed_sidecar(source, parent_dir=_PARENT, agent_id="a1", tool_use_id="call_1")
    _seed_sidecar(source, parent_dir="some-other-parent", agent_id="a1", tool_use_id="call_9")

    _write_parent(index, _parent_records([("call_1", "a1")], with_result=False), source_conn=source)
    child_id = _write_child(index, "a1", source_conn=source, raw_id=_seed_child_transcript(source, "a1"))

    link = _link(index, child_id)
    assert link["parent_tool_use_block_id"] == _tool_use_block_id(index, "call_1")
    assert link["method"] == "parent-tool-use-id"
    assert _dispatch_reason(link) is None


def test_sidecar_and_parent_result_disagreeing_is_a_contradiction(tmp_path: Path) -> None:
    index = _index_conn(tmp_path / "index.db")
    source = _source_conn(tmp_path / "source.db")
    _seed_sidecar(source, parent_dir=_PARENT, agent_id="a1", tool_use_id="call_2")

    _write_parent(index, _parent_records([("call_1", "a1"), ("call_2", "a2")]), source_conn=source)
    child_id = _write_child(index, "a1", source_conn=source, raw_id=_seed_child_transcript(source, "a1"))

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

    write_fixture_index_session(conn, _session("codex-parent"))
    child_id = write_fixture_index_session(conn, _session("codex-child", parent="codex-parent"))

    link = _link(conn, child_id)
    assert link["resolved_dst_session_id"] is not None
    assert link["parent_tool_use_block_id"] is None
    assert _dispatch_reason(link) == "origin-no-dispatch-identity"


def _seed_unrelated_claude_raws(source: sqlite3.Connection, count: int) -> None:
    """Other sessions' sidecars: rows an exact lookup never visits."""
    source.executemany(
        """INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms)
           VALUES (?, 'claude-code-session', ?, ?, 0, 0)""",
        [
            (
                f"raw-unrelated-{index}",
                _subagent_path(f"other-parent-{index}", "a1", ".meta.json"),
                hashlib.sha256(f"unrelated-{index}".encode()).digest(),
            )
            for index in range(count)
        ],
    )
    source.commit()


def _source_tier_steps_writing_child(tmp_path: Path, *, unrelated: int) -> int:
    """SQLite VM instructions the child's write runs on the source tier."""
    root = tmp_path / f"unrelated-{unrelated}"
    root.mkdir()
    index = _index_conn(root / "index.db")
    source = _source_conn(root / "source.db")
    _seed_unrelated_claude_raws(source, unrelated)
    _seed_sidecar(source, parent_dir=_PARENT, agent_id="a1", tool_use_id="call_1")
    _write_parent(index, _parent_records([("call_1", "a1")], with_result=False), source_conn=source)
    raw_id = _seed_child_transcript(source, "a1")
    steps = 0

    def _count() -> int:
        nonlocal steps
        steps += 1
        return 0

    source.set_progress_handler(_count, 1)
    try:
        child_id = _write_child(index, "a1", source_conn=source, raw_id=raw_id)
    finally:
        source.set_progress_handler(None, 1)
    assert _link(index, child_id)["parent_tool_use_block_id"] == _tool_use_block_id(index, "call_1")
    index.close()
    source.close()
    return steps


def test_sidecar_lookup_work_does_not_grow_with_the_archives_claude_raws(tmp_path: Path) -> None:
    """Binding one edge probes the child's own sidecar path, not every Claude raw.

    SQLite's progress handler counts executed VM instructions, so the two
    archives are compared exactly rather than by timing. Anti-vacuity: look
    the sidecar up by a leading-wildcard ``source_path LIKE`` (or filter it by
    ``origin`` through the origin index) and the count grows with the
    unrelated rows.
    """
    assert _source_tier_steps_writing_child(tmp_path, unrelated=400) == _source_tier_steps_writing_child(
        tmp_path, unrelated=4
    )


def test_retained_replay_binds_the_dispatch_the_sidecar_names(tmp_path: Path) -> None:
    """A replay of the same raws rebuilds the edge live ingest bound.

    The dispatch witness is the child's ``agent-*.meta.json`` sidecar in the
    source tier; the parent's result record is silent. Live ingest hands the
    writer its source handle. Anti-vacuity: drop ``source_conn`` from the
    retained-raw replay's writer call and the edge is rebuilt with no dispatch
    block and ``dispatch-evidence-absent``.
    """
    root = bootstrap_archive_root(tmp_path / "archive")
    parent_records = _parent_records([("call_1", "a1")], with_result=False)
    parent_path = f"/x/.claude/projects/proj/{_PARENT}.jsonl"
    child_path = _subagent_path(_PARENT, "a1", ".jsonl")

    def _jsonl(records: list[dict[str, object]]) -> bytes:
        return b"\n".join(json.dumps(record).encode() for record in records)

    with ArchiveStore.open_existing(root, read_only=False) as archive:

        def _retained(source_path: str, payload: bytes) -> str:
            return archive.write_raw_payload(
                provider=Provider.CLAUDE_CODE,
                payload=payload,
                source_path=source_path,
                canonical_source_path=source_path,
                acquired_at_ms=1,
            )

        _retained(parent_path, _jsonl(parent_records))
        _retained(_subagent_path(_PARENT, "a1", ".meta.json"), _sidecar_payload("call_1"))
        _retained(child_path, _jsonl(_child_records("a1")))

    replay_retained_components(root)

    with sqlite3.connect(root / "index.db") as index:
        edge = index.execute(
            """SELECT l.parent_tool_use_block_id, l.method, json_extract(l.evidence_json, '$.dispatch_reason'),
                      (SELECT b.block_id FROM blocks b WHERE b.tool_id = 'call_1' AND b.block_type = 'tool_use')
               FROM session_links l JOIN sessions s ON s.session_id = l.src_session_id
               WHERE s.native_id = ?""",
            # A Claude Code subagent's native id is scoped by its parent session.
            (f"{_PARENT}:agent-a1",),
        ).fetchall()

    assert len(edge) == 1
    block_id, method, reason, dispatch_block = tuple(edge[0])
    assert dispatch_block is not None
    assert (block_id, method, reason) == (dispatch_block, "parent-tool-use-id", None)


def test_retained_sidecar_remains_evidence_after_new_acquisition_moves(tmp_path: Path) -> None:
    from contextlib import closing

    with closing(_index_conn(tmp_path / "index.db")) as index, closing(_source_conn(tmp_path / "source.db")) as source:
        _seed_sidecar(source, parent_dir=_PARENT, agent_id="a1", tool_use_id="call_1")
        _write_parent(index, _parent_records([("call_1", "a1")], with_result=False), source_conn=source)
        old_raw = _seed_child_transcript(source, "a1")
        child_id = _write_child(index, "a1", source_conn=source, raw_id=old_raw)
        native_id = index.execute("SELECT native_id FROM sessions WHERE session_id = ?", (child_id,)).fetchone()[0]
        source.execute("UPDATE raw_sessions SET native_id = ? WHERE raw_id = ?", (native_id, old_raw))
        source.commit()
        new_path = _subagent_path(_PARENT, "a1", ".jsonl").replace("/proj/", "/moved/")
        new_raw = _seed_raw(
            source,
            raw_id="moved-child",
            source_path=new_path,
            payload=b"\n".join(json.dumps(row).encode() for row in _child_records("a1")),
        )
        _write_child(index, "a1", source_conn=source, raw_id=new_raw)
        edge = _link(index, child_id)
        assert edge["parent_tool_use_block_id"] == _tool_use_block_id(index, "call_1")
        assert _dispatch_reason(edge) is None
