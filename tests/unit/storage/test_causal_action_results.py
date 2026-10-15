"""Public actions retain every reply proved by actual message-parent evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.core.enums import ActionResultState, BlockType, ToolOutcome, ToolResultUnknownReason
from polylogue.sources.parsers.base import ParsedContentBlock
from polylogue.storage.sqlite.action_pairs import rebuild_all_action_pairs_sync
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.run_projection_relations import observed_event_relation_sql
from tests.infra.action_pairing import action_stream
from tests.infra.live_ingest import write_index_session


def test_proven_chain_preserves_ordered_unknown_and_error_results(tmp_path: Path) -> None:
    session = action_stream(
        "chain", [("call", "use", "shared", None), ("r1", "result", "shared", False), ("r2", "result", "shared", True)]
    )
    session.messages[1].blocks[0] = ParsedContentBlock(
        type=BlockType.TOOL_RESULT,
        tool_id="shared",
        text="r1",
        tool_outcome=ToolOutcome.UNKNOWN,
        outcome_unknown_reason=ToolResultUnknownReason.NOT_REPORTED,
    )
    session.messages[1].parent_message_provider_id = "call"
    session.messages[2].parent_message_provider_id = "r1"
    with ArchiveStore(tmp_path / "archive") as archive:
        session_id = write_index_session(archive, session)
        (action,) = archive.query_session_action_occurrences([session_id], limit=10)
        assert action.result_state == ActionResultState.OUTCOME_ERROR
        assert tuple(result.tool_outcome for result in action.results) == ("unknown", "error")
        assert tuple(result.output_text for result in action.results) == ("r1", "r2")
        assert action.output_text == "r1\nr2"
        event_sql = observed_event_relation_sql(source_where="1 = 1")
        events = archive._conn.execute(
            f"{event_sql} SELECT event_ref,status,evidence_refs_json,result_message_id FROM observed_events WHERE kind='tool_finished'"
        ).fetchall()
        assert len(events) == 1
        event = events[0]
        assert event["status"] == "failed"
        assert event["event_ref"] == f"observed-event:{action.tool_use_block_id}:tool_finished"
        assert len(json.loads(event["evidence_refs_json"])) == 3
        assert (
            event["result_message_id"]
            == archive._conn.execute(
                "SELECT message_id FROM messages WHERE session_id=? AND native_id='r2'", (session_id,)
            ).fetchone()[0]
        )
        rebuild_all_action_pairs_sync(archive._conn)
        assert archive.query_session_action_occurrences([session_id], limit=10) == [action]


def test_capability_snapshot_ignores_observation_time_but_keeps_evidence() -> None:
    from polylogue.archive.query.capability_catalog import capability_detail_page

    first = capability_detail_page(stats={"total_sessions": 4}, readiness={"state": "ready", "timestamp": 100})
    later = capability_detail_page(
        offset=1, stats={"total_sessions": 4}, readiness={"state": "ready", "timestamp": 200}
    )
    evidence = first["evidence"]
    later_evidence = later["evidence"]
    assert isinstance(evidence, dict) and isinstance(later_evidence, dict)
    assert evidence["readiness"]["timestamp"] == 100
    assert later_evidence["readiness"]["timestamp"] == 200
    snapshot, later_snapshot = first["snapshot"], later["snapshot"]
    assert isinstance(snapshot, dict) and isinstance(later_snapshot, dict)
    assert snapshot["id"] == later_snapshot["id"]


def test_visible_inherited_call_is_isolated_from_error_sibling(tmp_path: Path) -> None:
    from polylogue.core.enums import BranchType
    from polylogue.surfaces.payloads import ActionQueryRowPayload

    parent = action_stream("parent", [("call", "use", "shared", None)])
    success = action_stream("success", [("call", "use", "shared", None), ("ok", "result", "shared", False)])
    error = action_stream("error", [("call", "use", "shared", None), ("bad", "result", "shared", True)])
    for child in (success, error):
        child.parent_session_provider_id = "parent"
        child.branch_type = BranchType.FORK
        child.messages[1].parent_message_provider_id = "call"
    with ArchiveStore(tmp_path / "archive") as archive:
        ids = [write_index_session(archive, item) for item in (parent, success, error)]
        expected = (
            (ActionResultState.NO_RESULT, ()),
            (ActionResultState.OUTCOME_SUCCESS, ("ok",)),
            (ActionResultState.OUTCOME_ERROR, ("bad",)),
        )
        # One batch deliberately contains the same physical inherited call in three contexts.
        rows = archive.query_session_action_occurrences(ids, limit=10)
        for session_id, (state, texts) in zip(ids, expected, strict=True):
            (row,) = [row for row in rows if row.session_id == session_id]
            assert row.result_state == state
            assert tuple(item.output_text for item in row.results) == texts
            payload = ActionQueryRowPayload.from_row(row)
            assert tuple(item.output_text for item in payload.results) == texts
            for result in payload.results:
                assert (
                    archive._conn.execute(
                        "SELECT text FROM blocks WHERE block_id=?", (result.tool_result_block_id,)
                    ).fetchone()[0]
                    == result.output_text
                )
            assert "tool_result_block_id" not in payload.model_dump()
            assert "exit_code" not in payload.model_dump()


@pytest.mark.asyncio
async def test_python_action_facade_uses_causal_results_instead_of_fifo(tmp_path: Path) -> None:
    from polylogue import Polylogue

    session = action_stream(
        "api-chain",
        [("call", "use", "shared", None), ("ok", "result", "shared", False), ("bad", "result", "shared", True)],
    )
    session.messages[1].parent_message_provider_id = "call"
    session.messages[2].parent_message_provider_id = "ok"
    from tests.infra.archive_templates import run_off_event_loop

    def prepare() -> str:
        with ArchiveStore(tmp_path / "archive") as archive:
            return write_index_session(archive, session)

    session_id = run_off_event_loop(prepare)
    facade = Polylogue(archive_root=tmp_path / "archive", db_path=tmp_path / "archive" / "index.db")
    rows = await facade.get_actions_batch([session_id])
    (action,) = rows[session_id]
    assert action.result_state == ActionResultState.OUTCOME_ERROR
    assert action.tool_success is False
    assert tuple(item.output_text for item in action.results) == ("ok", "bad")
    # Action carries its existing collapsed search preview; results keep full bodies.
    assert action.output_text == "ok bad"


@pytest.mark.parametrize("mutation", ["block_delete", "message_delete"])
def test_removed_intermediate_recomputes_the_surviving_unique_stream(tmp_path: Path, mutation: str) -> None:
    session = action_stream(
        "revoked-chain",
        [("call", "use", "shared", None), ("r1", "result", "shared", False), ("r2", "result", "shared", True)],
    )
    session.messages[1].parent_message_provider_id = "call"
    session.messages[2].parent_message_provider_id = "r1"
    with ArchiveStore(tmp_path / "archive") as archive:
        session_id = write_index_session(archive, session)
        (original,) = archive.query_session_action_occurrences([session_id], limit=10)
        assert len(original.results) == 2
        if mutation == "block_delete":
            archive._conn.execute(
                "DELETE FROM blocks WHERE message_id IN (SELECT message_id FROM messages WHERE session_id=? AND native_id='r1')",
                (session_id,),
            )
        else:
            archive._conn.execute("DELETE FROM messages WHERE session_id=? AND native_id='r1'", (session_id,))
        (current,) = archive.query_session_action_occurrences([session_id], limit=10)
        # The surviving receipt can still be associated by a unique alternating
        # native-ID stream. It cannot retain a parent-proved plural chain.
        assert tuple(result.output_text for result in current.results) == ("r2",)
        assert current.result_state == ActionResultState.OUTCOME_ERROR
        rebuild_all_action_pairs_sync(archive._conn)
        assert archive.query_session_action_occurrences([session_id], limit=10) == [current]


def test_shared_association_cannot_borrow_an_excluded_intermediate_from_another_context() -> None:
    import sqlite3

    from polylogue.core.tool_association import tool_association_ctes_sql

    # The same original physical facts are present in two requested contexts;
    # only the full context includes the intermediate reply.
    messages = """association_messages(session_key,message_key,parent_key,role,message_position,variant_index) AS (
        VALUES ('full','call',NULL,'assistant',0,0),('full','middle','call','tool',1,0),
               ('full','tail','middle','tool',2,0),('cut','call',NULL,'assistant',0,0),
               ('cut','tail','middle','tool',1,0))"""
    blocks = """association_blocks(session_key,block_key,message_key,tool_id,is_use,outcome,unknown_reason,block_position) AS (
        VALUES ('full','use','call','shared',1,NULL,NULL,0),('full','reply1','middle','shared',0,'ok',NULL,0),
               ('full','reply2','tail','shared',0,'error',NULL,0),('cut','use','call','shared',1,NULL,NULL,0),
               ('cut','reply2','tail','shared',0,'error',NULL,0))"""
    with sqlite3.connect(":memory:") as conn:
        rows = conn.execute(
            f"WITH RECURSIVE {messages},{blocks},{tool_association_ctes_sql()} "
            "SELECT session_key,block_key,parent_proved FROM association_results ORDER BY session_key,block_key"
        ).fetchall()
    assert rows == [("cut", "reply2", 0), ("full", "reply1", 1), ("full", "reply2", 1)]


def test_nested_result_preview_preserves_full_length_and_clips_nul_text(tmp_path: Path) -> None:
    session = action_stream(
        "preview-chain",
        [("call", "use", "shared", None), ("a", "result", "shared", False), ("b", "result", "shared", True)],
    )
    session.messages[1].blocks[0].text = "a\0long"
    session.messages[1].parent_message_provider_id = "call"
    session.messages[2].blocks[0].text = "second"
    session.messages[2].parent_message_provider_id = "a"
    with ArchiveStore(tmp_path / "archive") as archive:
        session_id = write_index_session(archive, session)
        (row,) = archive.query_session_actions([session_id], limit=10, text_prefix_chars=1)
        assert [(item.output_text, item.output_text_chars) for item in row.results] == [("a", 6), ("s", 6)]


def test_mcp_root_payload_compacts_without_changing_the_json_document() -> None:
    from polylogue.mcp.payloads import MCPRootPayload

    data = {"items": [{"name": "neutral", "missing": None, "nested": [1, 2]}]}
    wire = MCPRootPayload(root=data).to_json()
    assert json.loads(wire) == data
    assert len(wire) < len(json.dumps(data, indent=2))
    assert "\n" not in wire
