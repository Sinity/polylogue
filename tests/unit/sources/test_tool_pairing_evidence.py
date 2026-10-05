"""Every tool call a provider declares reaches the actions relation intact.

The source of truth for these tests is ``devtools.tool_evidence_oracle``, which
reads the provider wire shapes directly and shares no code with the parsers. A
parser that loses an answer, renames its owner, duplicates it or rewrites its
outcome therefore disagrees with the oracle instead of moving both sides of the
comparison together.

Anti-vacuity: every conservation assertion is paired with a mutation of the
parsed session that must make the same oracle red. A test that passes both
before and after its mutation is proving nothing.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from devtools.tool_evidence_oracle import (
    ArchivedAction,
    declare_tool_evidence,
    judge_conservation,
)
from polylogue.analysis.tool_episodes import ToolEpisodeQuery
from polylogue.core.enums import BlockType, ToolOutcome
from polylogue.core.sources import origin_from_provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedSession
from polylogue.sources.parsers.chatgpt import parse as chatgpt_parse
from polylogue.sources.parsers.claude import parse_ai as claude_ai_parse
from polylogue.sources.tool_outcomes import derive_tool_outcomes
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.action_relation import action_relation_select_sql
from polylogue.storage.sqlite.archive_tiers import write as archive_write
from polylogue.storage.sqlite.archive_tiers.read_insights import ArchiveReadInsights
from polylogue.storage.sqlite.run_projection_relations import observed_event_relation_sql
from tests.infra.index_writer import close_fixture_index_connection, write_fixture_index_session
from tests.infra.live_ingest import write_session_sync

# ---------------------------------------------------------------------------
# Provider-shaped sources
# ---------------------------------------------------------------------------


def _chatgpt_node(
    node_id: str,
    *,
    parent: str | None,
    role: str,
    content: dict[str, object],
    recipient: str | None = None,
    status: str | None = None,
    name: str | None = None,
) -> dict[str, object]:
    author: dict[str, object] = {"role": role}
    if name:
        author["name"] = name
    message: dict[str, object] = {
        "id": node_id,
        "author": author,
        "content": content,
        "create_time": 1_717_430_400.0,
    }
    if recipient:
        message["recipient"] = recipient
    if status:
        message["status"] = status
    node: dict[str, object] = {"id": node_id, "message": message}
    if parent:
        node["parent"] = parent
    return node


def chatgpt_conversation() -> dict[str, object]:
    """One conversation covering every ChatGPT tool-answer shape.

    ``python`` succeeds, ``container.exec`` fails, ``file_search.msearch``
    answers with a plain-text node whose status concludes nothing, ``browser``
    answers with two chained retrieval nodes, and ``computer.do`` is abandoned
    with no answer at all.
    """
    mapping: dict[str, object] = {
        "root": {"id": "root", "parent": None, "message": None},
        "u1": _chatgpt_node("u1", parent="root", role="user", content={"content_type": "text", "parts": ["go"]}),
        "call-ok": _chatgpt_node(
            "call-ok",
            parent="u1",
            role="assistant",
            recipient="python",
            content={"content_type": "code", "text": "print(1)"},
        ),
        "res-ok": _chatgpt_node(
            "res-ok",
            parent="call-ok",
            role="tool",
            name="python",
            status="finished_successfully",
            content={"content_type": "execution_output", "text": "1"},
        ),
        "call-err": _chatgpt_node(
            "call-err",
            parent="res-ok",
            role="assistant",
            recipient="container.exec",
            content={"content_type": "code", "text": "false"},
        ),
        "res-err": _chatgpt_node(
            "res-err",
            parent="call-err",
            role="tool",
            name="container.exec",
            status="finished_successfully",
            content={"content_type": "system_error", "name": "ExecError", "text": "exit 1"},
        ),
        "call-unknown": _chatgpt_node(
            "call-unknown",
            parent="res-err",
            role="assistant",
            recipient="file_search.msearch",
            content={"content_type": "code", "text": '{"queries":["x"]}'},
        ),
        "res-unknown": _chatgpt_node(
            "res-unknown",
            parent="call-unknown",
            role="tool",
            name="file_search",
            status="in_progress",
            content={"content_type": "text", "parts": ["one hit"]},
        ),
        "call-fanout": _chatgpt_node(
            "call-fanout",
            parent="res-unknown",
            role="assistant",
            recipient="browser",
            content={"content_type": "code", "text": "search()"},
        ),
        "res-fanout-1": _chatgpt_node(
            "res-fanout-1",
            parent="call-fanout",
            role="tool",
            name="browser",
            status="finished_successfully",
            content={"content_type": "tether_browsing_display", "result": "page listing"},
        ),
        "res-fanout-2": _chatgpt_node(
            "res-fanout-2",
            parent="res-fanout-1",
            role="tool",
            name="browser",
            status="finished_successfully",
            content={"content_type": "tether_quote", "text": "quoted excerpt", "domain": "example.test"},
        ),
        "call-abandoned": _chatgpt_node(
            "call-abandoned",
            parent="res-fanout-2",
            role="assistant",
            recipient="computer.do",
            content={"content_type": "code", "text": "click()"},
        ),
    }
    return {"id": "conv-tool-evidence", "conversation_id": "conv-tool-evidence", "mapping": mapping}


def claude_web_conversation() -> dict[str, object]:
    """A Claude web export: id-less tool calls, one abandoned, one unowned answer."""
    return {
        "uuid": "claude-web-tool-evidence",
        "name": "tool evidence",
        "chat_messages": [
            {"uuid": "m0", "sender": "human", "text": "go", "content": [{"type": "text", "text": "go"}]},
            {
                "uuid": "m1",
                "sender": "assistant",
                "text": "",
                "content": [
                    {"type": "tool_use", "name": "repl", "input": {"code": "1+1"}},
                    {"type": "tool_result", "name": "repl", "is_error": False, "content": "2"},
                    {"type": "tool_use", "name": "web_search", "input": {"query": "x"}},
                    {"type": "tool_result", "name": "web_search", "is_error": True, "content": "boom"},
                    {"type": "tool_use", "name": "artifacts", "input": {"id": "a"}},
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Archive read-back
# ---------------------------------------------------------------------------


def _archived_actions(db_path: Path, session_id: str) -> list[ArchivedAction]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT tool_use_block_id, tool_name, result_state, tool_result_block_id,
                   (SELECT tool_id FROM blocks WHERE block_id = actions.tool_use_block_id) AS tool_id
            FROM actions
            WHERE session_id = ?
            ORDER BY tool_use_block_id
            """,
            (session_id,),
        ).fetchall()
    finally:
        conn.close()
    return [
        ArchivedAction(
            tool_use_block_id=str(row["tool_use_block_id"]),
            tool_id=row["tool_id"],
            tool_name=row["tool_name"],
            result_state=str(row["result_state"]),
            tool_result_block_id=row["tool_result_block_id"],
        )
        for row in rows
    ]


def _physical_results(db_path: Path, session_id: str) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM blocks WHERE session_id = ? AND block_type = 'tool_result'",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    return int(row[0]) if row else 0


def _write_and_read(db_path: Path, session: ParsedSession) -> tuple[list[ArchivedAction], int]:
    """Write through the canonical writer; read back the actions and the stored answers."""
    session_id = write_session_sync(db_path, session)
    return _archived_actions(db_path, session_id), _physical_results(db_path, session_id)


# ---------------------------------------------------------------------------
# Conservation through the production route
# ---------------------------------------------------------------------------


def test_chatgpt_tool_evidence_reaches_the_actions_relation(test_db: Path) -> None:
    payload = chatgpt_conversation()
    evidence = declare_tool_evidence(payload, origin="chatgpt-export")
    actions, stored_results = _write_and_read(test_db, chatgpt_parse(payload, "conv-tool-evidence"))

    verdict = judge_conservation(evidence, actions, physical_results=stored_results)
    assert verdict.conserved, verdict.to_dict()

    states = {action.tool_name: action.result_state for action in actions}
    assert states["python"] == "outcome_success"
    assert states["container.exec"] == "outcome_error"
    assert states["file_search.msearch"] == "outcome_unknown"
    assert states["browser"] == "outcome_success"
    assert states["computer.do"] == "no_result"


def test_claude_web_idless_tool_pairs_reach_the_actions_relation(test_db: Path) -> None:
    payload = claude_web_conversation()
    evidence = declare_tool_evidence(payload, origin="claude-ai-export")
    actions, stored_results = _write_and_read(test_db, claude_ai_parse(payload, "claude-web-tool-evidence"))

    verdict = judge_conservation(evidence, actions, physical_results=stored_results)
    assert verdict.conserved, verdict.to_dict()

    states = {action.tool_name: action.result_state for action in actions}
    assert states["repl"] == "outcome_success"
    assert states["web_search"] == "outcome_error"
    # The abandoned call keeps its own structural identity rather than
    # borrowing the answered call's, so it stays queryable as itself.
    assert states["artifacts"] == "no_result"
    identities = {action.tool_name: action.tool_id for action in actions}
    assert len({identity for identity in identities.values() if identity}) == 3
    assert all(identity.startswith("structural:claude-web:") for identity in identities.values() if identity)


def test_claude_web_unowned_tool_result_is_not_given_a_borrowed_identity(test_db: Path) -> None:
    """A result with no preceding unanswered call stays unowned, never reassigned."""
    payload = {
        "uuid": "claude-web-unowned",
        "chat_messages": [
            {
                "uuid": "m1",
                "sender": "assistant",
                "text": "",
                "content": [
                    {"type": "tool_use", "name": "repl", "input": {"code": "1"}},
                    {"type": "tool_result", "name": "repl", "is_error": False, "content": "1"},
                    {"type": "tool_result", "name": "repl", "is_error": False, "content": "stray"},
                ],
            }
        ],
    }
    session = claude_ai_parse(payload, "claude-web-unowned")
    results = [block for message in session.messages for block in message.blocks if block.type is BlockType.TOOL_RESULT]
    assert len(results) == 2
    assert results[0].tool_id is not None
    assert results[1].tool_id is None

    actions, stored_results = _write_and_read(test_db, session)
    assert [action.result_state for action in actions] == ["outcome_success"]
    assert stored_results == 2


def test_chatgpt_fanout_answer_keeps_the_calling_node_as_owner() -> None:
    """A chained retrieval node names the call, never the previous answer."""
    session = chatgpt_parse(chatgpt_conversation(), "conv-tool-evidence")
    owners = {
        block.tool_id
        for message in session.messages
        for block in message.blocks
        if block.type is BlockType.TOOL_RESULT and (block.metadata or {}).get("content_type") == "tether_quote"
    }
    assert owners == {"call-fanout"}


def test_chatgpt_abandoned_call_is_absent_not_unknown(test_db: Path) -> None:
    """Legitimate absence and an unreadable outcome are different states."""
    actions, _stored = _write_and_read(test_db, chatgpt_parse(chatgpt_conversation(), "conv-tool-evidence"))
    by_name = {action.tool_name: action for action in actions}
    assert by_name["computer.do"].result_state == "no_result"
    assert by_name["computer.do"].tool_result_block_id is None
    assert by_name["file_search.msearch"].result_state == "outcome_unknown"
    assert by_name["file_search.msearch"].tool_result_block_id is not None


# ---------------------------------------------------------------------------
# Mutations the oracle must catch
# ---------------------------------------------------------------------------


def _map_blocks(
    session: ParsedSession,
    transform: Callable[[ParsedContentBlock], Sequence[ParsedContentBlock]],
) -> ParsedSession:
    messages = []
    for message in session.messages:
        blocks: list[ParsedContentBlock] = []
        for block in message.blocks:
            blocks.extend(transform(block))
        messages.append(message.model_copy(update={"blocks": blocks}))
    return session.model_copy(update={"messages": messages})


def _drop_one_result(session: ParsedSession) -> ParsedSession:
    dropped = False

    def transform(block: ParsedContentBlock) -> Sequence[ParsedContentBlock]:
        nonlocal dropped
        if not dropped and block.type is BlockType.TOOL_RESULT:
            dropped = True
            return ()
        return (block,)

    mutated = _map_blocks(session, transform)
    assert dropped
    return mutated


def _lose_one_owner(session: ParsedSession) -> ParsedSession:
    cleared = False

    def transform(block: ParsedContentBlock) -> Sequence[ParsedContentBlock]:
        nonlocal cleared
        if not cleared and block.type is BlockType.TOOL_RESULT:
            cleared = True
            return (block.model_copy(update={"tool_id": None}),)
        return (block,)

    mutated = _map_blocks(session, transform)
    assert cleared
    return mutated


def _mismatch_one_id(session: ParsedSession) -> ParsedSession:
    changed = False

    def transform(block: ParsedContentBlock) -> Sequence[ParsedContentBlock]:
        nonlocal changed
        if not changed and block.type is BlockType.TOOL_RESULT and block.tool_id:
            changed = True
            return (block.model_copy(update={"tool_id": f"{block.tool_id}-elsewhere"}),)
        return (block,)

    mutated = _map_blocks(session, transform)
    assert changed
    return mutated


def _duplicate_one_result(session: ParsedSession) -> ParsedSession:
    duplicated = False

    def transform(block: ParsedContentBlock) -> Sequence[ParsedContentBlock]:
        nonlocal duplicated
        if not duplicated and block.type is BlockType.TOOL_RESULT and block.tool_id:
            duplicated = True
            return (block, block.model_copy())
        return (block,)

    mutated = _map_blocks(session, transform)
    assert duplicated
    return mutated


def _flip_one_outcome(session: ParsedSession) -> ParsedSession:
    flipped = False

    def transform(block: ParsedContentBlock) -> Sequence[ParsedContentBlock]:
        nonlocal flipped
        if not flipped and block.type is BlockType.TOOL_RESULT and isinstance(block.is_error, bool):
            flipped = True
            return (block.model_copy(update={"is_error": not block.is_error, "tool_outcome": None}),)
        return (block,)

    mutated = _map_blocks(session, transform)
    assert flipped
    return mutated


_MUTATIONS: tuple[tuple[str, Callable[[ParsedSession], ParsedSession], str], ...] = (
    ("dropped-result", _drop_one_result, "lost_results"),
    ("owner-loss", _lose_one_owner, "lost_results"),
    ("id-mismatch", _mismatch_one_id, "lost_results"),
    ("duplicate-result", _duplicate_one_result, "duplicated_results"),
    ("outcome-field", _flip_one_outcome, "outcome_disagreements"),
)


@pytest.mark.parametrize(("label", "mutate", "expected_finding"), _MUTATIONS, ids=[case[0] for case in _MUTATIONS])
def test_source_to_action_oracle_refuses_a_mutated_pairing(
    tmp_path: Path,
    label: str,
    mutate: Callable[[ParsedSession], ParsedSession],
    expected_finding: str,
) -> None:
    """Each controlled mutation of the archived pairing turns the oracle red.

    The unmutated write of the same source is asserted conserved first, so a
    mutation that changes nothing cannot pass this test.
    """
    payload = chatgpt_conversation()
    evidence = declare_tool_evidence(payload, origin="chatgpt-export")
    session = chatgpt_parse(payload, "conv-tool-evidence")

    from polylogue.storage.sqlite.connection import open_connection

    # Each write owns its own archive root: one root has one active Index.
    clean_db = tmp_path / f"clean-{label}" / "index.db"
    clean_db.parent.mkdir()
    with open_connection(clean_db):
        pass
    clean_actions, clean_results = _write_and_read(clean_db, session)
    baseline = judge_conservation(evidence, clean_actions, physical_results=clean_results)
    assert baseline.conserved, baseline.to_dict()

    mutated_db = tmp_path / f"mutated-{label}" / "index.db"
    mutated_db.parent.mkdir()
    with open_connection(mutated_db):
        pass
    mutated_actions, mutated_results = _write_and_read(mutated_db, mutate(session))
    verdict = judge_conservation(evidence, mutated_actions, physical_results=mutated_results)
    assert not verdict.conserved, f"{label} left the oracle green"
    assert getattr(verdict, expected_finding), verdict.to_dict()


def test_oracle_reads_an_interrupted_claude_code_tail(tmp_path: Path) -> None:
    """A transcript that ends on an unanswered call declares that call absent."""
    records = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_answered", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_answered", "is_error": False, "content": "ok"},
                ],
            },
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_interrupted", "name": "Bash", "input": {"command": "sleep"}},
                ],
            },
        },
    ]
    evidence = declare_tool_evidence(records, origin="claude-code-session")
    outcomes = {call.key: call.outcome for call in evidence.calls}
    assert outcomes == {"toolu_answered": "ok", "toolu_interrupted": "no_result"}


def test_oracle_reads_a_codex_resultless_construct() -> None:
    """`web_search_call` declares a call the wire never answers."""
    records = [
        {"type": "response_item", "payload": {"type": "web_search_call", "status": "completed", "action": {}}},
        {
            "type": "response_item",
            "payload": {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{}"},
        },
        {
            "type": "response_item",
            "payload": {"type": "function_call_output", "call_id": "call_1", "output": {"exit_code": 0}},
        },
    ]
    evidence = declare_tool_evidence(records, origin="codex-session")
    outcomes = {call.tool_name: call.outcome for call in evidence.calls}
    assert outcomes == {"web_search_call": "no_result", "shell": "ok"}
    assert evidence.result_records == 1


def test_oracle_counts_a_result_naming_no_declared_call() -> None:
    """A physical answer whose call the source never states is an orphan, not a pair."""
    records = [
        {
            "type": "response_item",
            "payload": {"type": "function_call_output", "call_id": "call_missing", "output": {"exit_code": 0}},
        },
    ]
    evidence = declare_tool_evidence(records, origin="codex-session")
    assert evidence.calls == ()
    assert evidence.orphan_results == 1


@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize("reported", ["ok", "error", "unknown"])
@pytest.mark.parametrize("parent_proved", [True, False])
def test_parent_proven_fanout_preserves_all_results_and_aggregate_verdict(
    test_db: Path, reported: str, parent_proved: bool, append: bool
) -> None:
    payload = chatgpt_conversation()
    mapping = payload["mapping"]
    assert isinstance(mapping, dict)
    second = mapping["res-fanout-2"]
    assert isinstance(second, dict)
    wire = second["message"]
    assert isinstance(wire, dict)
    if reported == "error":
        wire["content"] = {"content_type": "system_error", "name": "ExecError", "text": "neutral failed reply"}
    elif reported == "unknown":
        wire["content"] = {"content_type": "text", "parts": ["neutral incomplete reply"]}
        wire["status"] = "in_progress"
    parsed = chatgpt_parse(payload, "conv-tool-evidence")
    if not parent_proved:
        # Keep the complete physical replies and reusable tool ID, while
        # removing only the native per-invocation proof from model-backed input.
        parsed = parsed.model_copy(
            update={
                "messages": [
                    message.model_copy(update={"parent_message_provider_id": None})
                    if message.provider_message_id in {"res-fanout-1", "res-fanout-2"}
                    else message
                    for message in parsed.messages
                ]
            }
        )
    reduced = derive_tool_outcomes(
        list(parsed.messages), parsed.session_events, origin=origin_from_provider(parsed.source_name)
    )
    browser_use = next(
        block
        for message in reduced
        for block in message.blocks
        if block.type is BlockType.TOOL_USE and block.tool_name == "browser"
    )
    expected = reported if parent_proved else "unknown"
    assert browser_use.tool_outcome is ToolOutcome(expected)
    if append:
        appended_ids = {"res-fanout-2", "call-abandoned"}
        baseline = parsed.model_copy(
            update={"messages": [m for m in parsed.messages if m.provider_message_id not in appended_ids]}
        )
        # A merge-append write carries only the new records; the later reply
        # must still re-decide the earlier invocation's verdict.
        tail = parsed.model_copy(
            update={"messages": [m for m in parsed.messages if m.provider_message_id in appended_ids]}
        )
        write_session_sync(test_db, baseline)
        conn = connect_measured(test_db)
        conn.row_factory = sqlite3.Row
        try:
            sid = write_fixture_index_session(conn, tail, merge_append=True, content_hash=session_content_hash(parsed))
        finally:
            close_fixture_index_connection(conn)
    else:
        sid = write_session_sync(test_db, parsed)
    assert _physical_results(test_db, sid) == 5
    with sqlite3.connect(f"file:{test_db}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        generic = conn.execute(action_relation_select_sql() + " ORDER BY tool_use_block_id").fetchall()
        cached = conn.execute("SELECT * FROM action_pairs ORDER BY tool_use_block_id").fetchall()
        assert [(row["tool_use_block_id"], row["tool_outcome"], row["tool_result_block_id"]) for row in generic] == [
            (row["tool_use_block_id"], row["tool_outcome"], row["tool_result_block_id"]) for row in cached
        ]
        browser = next(row for row in generic if row["tool_name"] == "browser")
        # A later reply changes the earlier invocation's block verdict. Its
        # message hash must be the original row producer's current digest.
        message = conn.execute(
            f"SELECT {', '.join(archive_write._REHASH_MESSAGE_COLUMNS)}, content_hash FROM messages WHERE message_id=?",
            (browser["message_id"],),
        ).fetchone()
        assert message is not None
        message_index = {name: index for index, name in enumerate(archive_write._REHASH_MESSAGE_COLUMNS)}
        block_index = {name: index for index, name in enumerate(archive_write._REHASH_BLOCK_COLUMNS)}
        stored_blocks = conn.execute(
            f"SELECT {', '.join(archive_write._REHASH_BLOCK_COLUMNS)} FROM blocks WHERE message_id=? ORDER BY position",
            (browser["message_id"],),
        )
        expected_message_hash = archive_write._message_row_hash(
            sid,
            message["native_id"],
            int(message["position"]),
            int(message["variant_index"] or 0),
            archive_write._row_fields_digest(message, message_index),
            archive_write._stored_block_hash_parts(stored_blocks, block_index),
        )
        assert message["content_hash"] == expected_message_hash
        assert browser["tool_outcome"] == expected
        assert browser["tool_result_block_id"] is None
        assert browser["output_text"] is None
        assert browser["exit_code"] is None
        events = conn.execute(
            observed_event_relation_sql(source_where="1=1")
            + " SELECT * FROM observed_events WHERE kind='tool_finished' AND tool_name='browser'"
        ).fetchall()
        reader = ArchiveReadInsights(
            conn,
            checkpoint=lambda: None,
            normalize_origin=lambda value: value,
            iso_from_milliseconds=lambda value: str(value),
        )
        episodes = reader.list_tool_episode_insights(ToolEpisodeQuery(session_id=sid, tool="browser"))
        assert len(episodes) == 1
        assert episodes[0].result_output is None
        if parent_proved:
            assert reader.degraded_components == ("tool_episode_plural_output_omitted",)
            assert episodes[0].tool_result_block_id is None
            assert episodes[0].is_error is None and episodes[0].exit_code is None
            assert len(events) == 1
            assert events[0]["status"] == ("failed" if reported == "error" else reported)
            refs = json.loads(events[0]["evidence_refs_json"])
            assert len(refs) == 3 and len(set(refs)) == 3
            assert events[0]["result_message_id"] is None and events[0]["result_position"] is None
        else:
            assert reader.degraded_components == ()
            assert events == []


@pytest.mark.parametrize("alternatives", [False, True])
def test_native_reply_siblings_distinguish_alternatives_from_sequential_fanout(
    test_db: Path, alternatives: bool
) -> None:
    payload = chatgpt_conversation()
    mapping = payload["mapping"]
    assert isinstance(mapping, dict)
    first = mapping["res-fanout-1"]
    second = mapping["res-fanout-2"]
    assert isinstance(first, dict) and isinstance(second, dict)
    invocation = first["parent"]
    assert isinstance(invocation, str)
    if alternatives:
        second["parent"] = invocation
        call = mapping[invocation]
        assert isinstance(call, dict)
        call["children"] = ["res-fanout-1", "res-fanout-2"]
    parsed = chatgpt_parse(payload, "conv-tool-evidence")
    replies = [
        message for message in parsed.messages if message.provider_message_id in {"res-fanout-1", "res-fanout-2"}
    ]
    assert len(replies) == 2
    if alternatives:
        assert replies[0].parent_message_provider_id == replies[1].parent_message_provider_id
        assert replies[0].variant_index != replies[1].variant_index
    else:
        assert replies[1].parent_message_provider_id == replies[0].provider_message_id
    expected = ToolOutcome.UNKNOWN if alternatives else ToolOutcome.OK
    reduced = derive_tool_outcomes(
        list(parsed.messages), parsed.session_events, origin=origin_from_provider(parsed.source_name)
    )
    browser = next(
        block
        for message in reduced
        for block in message.blocks
        if block.type is BlockType.TOOL_USE and block.tool_name == "browser"
    )
    assert browser.tool_outcome is expected
    sid = write_session_sync(test_db, parsed)
    assert _physical_results(test_db, sid) == 5
    with sqlite3.connect(f"file:{test_db}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        action = next(row for row in conn.execute(action_relation_select_sql()) if row["tool_name"] == "browser")
        assert action["tool_outcome"] == expected.value
        assert action["tool_result_block_id"] is None
