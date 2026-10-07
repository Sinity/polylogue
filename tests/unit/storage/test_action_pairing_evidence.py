"""Association evidence is causal, ambiguity-visible and shared by every reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.query.expression import parse_unit_source_expression
from polylogue.storage.sqlite.action_pairs import rebuild_all_action_pairs_sync, refresh_action_pairs
from polylogue.storage.sqlite.action_relation import action_relation_select_sql
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.run_projection_relations import observed_event_relation_sql
from tests.infra.action_pairing import action_stream
from tests.infra.live_ingest import write_index_session


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("u0 r0", [("u0", "r0", "ok")]),
        ("old u0 bad", [("u0", "bad", "error")]),
        ("old older u0 bad u1 r1", [("u0", "bad", "error"), ("u1", "r1", "ok")]),
        ("foreign u0 bad", [("u0", "bad", "error")]),
        ("u0 r0 u1 bad", [("u0", "r0", "ok"), ("u1", "bad", "error")]),
        ("u0", [("u0", None, "no_result")]),
        ("u0 r0 u1", [("u0", "r0", "ok"), ("u1", None, "no_result")]),
        ("u0 u1 bad", [("u0", None, "unknown"), ("u1", None, "unknown")]),
        ("u0 r0 u1 u2 bad", [("u0", "r0", "ok"), ("u1", None, "unknown"), ("u2", None, "unknown")]),
        ("u0 r0 duplicate u1 bad", [("u0", None, "unknown"), ("u1", None, "unknown")]),
        ("u0 u1 bad r0", [("u0", None, "unknown"), ("u1", None, "unknown")]),
        ("old older", []),
    ],
)
def test_writer_canonical_bulk_and_events_agree_on_causal_evidence(
    tmp_path: Path, sequence: str, expected: list[tuple[str, str | None, str]]
) -> None:
    """Causal pairing rejects shifted receipts and ambiguous parallel reuse."""
    events = [
        (
            name,
            "use" if name.startswith("u") else "result",
            "different" if name == "foreign" else "reused",
            name == "bad",
        )
        for name in sequence.split()
    ]
    with ArchiveStore(tmp_path / "archive") as archive:
        session_id = write_index_session(archive, action_stream("evidence", events))
        conn = archive._conn
        block_ids = {
            row["native_id"]: row["block_id"]
            for row in conn.execute(
                "SELECT m.native_id, b.block_id FROM blocks b JOIN messages m ON m.message_id = b.message_id WHERE b.session_id = ?",
                (session_id,),
            )
        }
        expected_rows = [
            (
                name,
                block_ids[result] if result else None,
                outcome,
                "no_result"
                if outcome == "no_result"
                else "outcome_success"
                if outcome == "ok"
                else f"outcome_{outcome}",
                "ambiguous_tool_id_reuse" if outcome == "unknown" else None,
            )
            for name, result, outcome in expected
        ]
        materialized_columns = "tool_command, tool_result_block_id, tool_outcome, outcome_unknown_reason"
        materialized_sql = f"SELECT {materialized_columns} FROM action_pairs WHERE session_id = ? ORDER BY tool_command"
        materialized_expected = [
            (
                name,
                result_block,
                {"ok": "ok", "error": "error", "unknown": "unknown", "no_result": "no_result"}[outcome],
                reason,
            )
            for name, result_block, outcome, _state, reason in expected_rows
        ]
        assert [tuple(row) for row in conn.execute(materialized_sql, (session_id,))] == materialized_expected
        for bounded in (False, True):
            sql = action_relation_select_sql(session_placeholders="?" if bounded else None)
            parameters = (session_id,) * 3 if bounded else ()
            columns = "tool_command, tool_result_block_id, result_state, outcome_unknown_reason"
            rows = conn.execute(f"SELECT {columns} FROM ({sql}) ORDER BY tool_command", parameters).fetchall()
            canonical_expected = [
                (name, result_block, state, reason) for name, result_block, _outcome, state, reason in expected_rows
            ]
            assert [tuple(row) for row in rows] == canonical_expected
            for row in conn.execute(f"SELECT * FROM ({sql})", parameters):
                if row["tool_result_block_id"] is None:
                    assert row["output_text"] is None
                    assert row["is_error"] is None
                    assert row["exit_code"] is None

        # The observed-event owner must compute the same links independently of
        # whether the action_pairs cache has been refreshed.
        conn.execute("DELETE FROM action_pairs WHERE session_id = ?", (session_id,))
        expected_events = sorted(
            (f"observed-event:{block_ids[name]}:tool_finished", "failed" if outcome == "error" else "ok")
            for name, result, outcome in expected
            if result is not None
        )
        for bounded in (False, True):
            ctes = observed_event_relation_sql(source_where="1", session_scoped=bounded)
            event_parameters = (session_id,) * 2 if bounded else ()
            event_rows = conn.execute(
                ctes + " SELECT event_ref, status FROM observed_events WHERE kind = 'tool_finished' ORDER BY event_ref",
                event_parameters,
            ).fetchall()
            assert [tuple(row) for row in event_rows] == expected_events

        refresh_action_pairs(conn, session_id)
        assert [tuple(row) for row in conn.execute(materialized_sql, (session_id,))] == materialized_expected
        rebuild_all_action_pairs_sync(conn)
        assert [tuple(row) for row in conn.execute(materialized_sql, (session_id,))] == materialized_expected
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        conn.commit()
    with ArchiveStore.open_existing(tmp_path / "archive") as reopened:
        assert [tuple(row) for row in reopened._conn.execute(materialized_sql, (session_id,))] == materialized_expected


def test_branch_sessions_and_different_tool_ids_do_not_share_ambiguity(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path / "archive") as archive:
        broken = action_stream(
            "fork-a",
            [
                ("a", "use", "same", None),
                ("b", "use", "same", None),
                ("r", "result", "same", False),
                ("c", "use", "separate", None),
                ("s", "result", "separate", True),
            ],
        )
        first = write_index_session(archive, broken)
        second = write_index_session(
            archive, action_stream("fork-b", [("a", "use", "same", None), ("r", "result", "same", False)])
        )
        rows = archive._conn.execute(
            "SELECT session_id, tool_command, result_state FROM actions ORDER BY session_id, tool_command"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            (first, "a", "outcome_unknown"),
            (first, "b", "outcome_unknown"),
            (first, "c", "outcome_error"),
            (second, "a", "outcome_success"),
        ]


@pytest.mark.parametrize("result_position", [0, 1])
def test_variant_order_cannot_certify_a_cross_branch_use(tmp_path: Path, result_position: int) -> None:
    session = action_stream("variants", [("u", "use", "same", None), ("r", "result", "same", False)])
    session.messages[0].position = 0
    session.messages[0].variant_index = 0
    session.messages[1].position = result_position
    session.messages[1].variant_index = 1
    with ArchiveStore(tmp_path / "archive") as archive:
        write_index_session(archive, session)
        rows = archive._conn.execute(
            "SELECT tool_result_block_id, result_state, outcome_unknown_reason FROM actions"
        ).fetchall()
        assert [tuple(row) for row in rows] == [(None, "outcome_unknown", "ambiguous_tool_id_reuse")]


def test_complete_reingest_resolves_ambiguity_at_the_ordinary_write_boundary(tmp_path: Path) -> None:
    incomplete = action_stream(
        "reingest",
        [("u0", "use", "same", None), ("u1", "use", "same", None), ("r1", "result", "same", False)],
    )
    complete = action_stream(
        "reingest",
        [
            ("u0", "use", "same", None),
            ("r0", "result", "same", True),
            ("u1", "use", "same", None),
            ("r1", "result", "same", False),
        ],
    )
    with ArchiveStore(tmp_path / "archive") as archive:
        session_id = write_index_session(archive, incomplete)
        before = archive._conn.execute("SELECT result_state FROM actions ORDER BY tool_command").fetchall()
        assert [row[0] for row in before] == ["outcome_unknown", "outcome_unknown"]
        assert write_index_session(archive, complete) == session_id
        rows = archive._conn.execute("SELECT output_text, result_state FROM actions ORDER BY tool_command").fetchall()
        assert [tuple(row) for row in rows] == [("r0", "outcome_error"), ("r1", "outcome_success")]


@pytest.mark.parametrize("reason", ["not_reported", "distrusted"])
def test_action_read_routes_preserve_paired_unknown_reason(tmp_path: Path, reason: str) -> None:
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock
    from polylogue.surfaces.payloads import ActionQueryRowPayload

    session = action_stream("unknown-reason", [("u", "use", "same", None), ("r", "result", "same", False)])
    session.source_name = Provider.CLAUDE_CODE
    session.messages[1].blocks[0] = ParsedContentBlock(
        type=BlockType.TOOL_RESULT, tool_id="same", text="r", outcome_unknown_reason=reason
    )
    source = parse_unit_source_expression("actions where tool:Bash")
    assert source is not None
    with ArchiveStore(tmp_path / "archive") as archive:
        session_id = write_index_session(archive, session)
        routes = (
            archive.query_actions(source.predicate, limit=1),
            archive.query_session_actions([session_id], limit=1, text_prefix_chars=1),
            archive.query_session_action_occurrences([session_id], limit=1),
        )
        for rows in routes:
            assert len(rows) == 1
            row = rows[0]
            assert row.result_state == "outcome_unknown"
            assert row.tool_result_block_id is not None
            assert row.outcome_unknown_reason == reason
            payload = ActionQueryRowPayload.from_row(row)
            assert payload.outcome_unknown_reason == reason
            assert payload.model_dump(mode="json")["outcome_unknown_reason"] == reason
        episode = archive.list_tool_episode_insights()[0]
        assert episode.tool_result_block_id is not None
        assert episode.result_state == "outcome_unknown"
        assert episode.outcome_unknown_reason == reason
        assert episode.caveat == "outcome unknown: paired structural result has no trusted verdict"


def test_action_unknown_association_and_missing_result_remain_distinct(tmp_path: Path) -> None:
    session = action_stream(
        "unknown-association",
        [
            ("u0", "use", "same", None),
            ("u1", "use", "same", None),
            ("r", "result", "same", False),
            ("missing", "use", "other", None),
        ],
    )
    source = parse_unit_source_expression("actions where tool:Bash")
    assert source is not None
    with ArchiveStore(tmp_path / "archive") as archive:
        write_index_session(archive, session)
        rows = archive.query_actions(source.predicate)
        assert {row.tool_command: row.outcome_unknown_reason for row in rows} == {
            "u0": "ambiguous_tool_id_reuse",
            "u1": "ambiguous_tool_id_reuse",
            "missing": None,
        }
        episodes = archive.list_tool_episode_insights()
        assert sorted(episode.caveat for episode in episodes) == [
            "outcome unknown: ambiguous result association",
            "outcome unknown: ambiguous result association",
            "outcome unknown: no paired structural result",
        ]


@pytest.mark.parametrize("tool_id", ["", "paired"])
def test_append_reconciliation_does_not_pair_empty_tool_ids(tmp_path: Path, tool_id: str) -> None:
    from polylogue.storage.sqlite.archive_tiers.write import _reconcile_tool_use_outcomes

    session = action_stream("empty-id", [("u", "use", tool_id, None), ("r", "result", tool_id, False)])
    source = parse_unit_source_expression("actions where tool:Bash")
    assert source is not None
    with ArchiveStore(tmp_path / "archive") as archive:
        session_id = write_index_session(archive, session)
        _reconcile_tool_use_outcomes(archive._conn, session_id)
        use = archive._conn.execute(
            "SELECT tool_outcome FROM blocks WHERE session_id = ? AND block_type = 'tool_use'", (session_id,)
        ).fetchone()
        assert use[0] == ("no_result" if tool_id == "" else "ok")
        action = archive.query_actions(source.predicate)[0]
        assert action.result_state == ("no_result" if tool_id == "" else "outcome_success")
        assert (action.tool_result_block_id is None) == (tool_id == "")


def test_episode_context_preserves_composed_message_boundaries_and_result_anchor(tmp_path: Path) -> None:
    from polylogue.analysis.tool_episodes import ToolEpisodeQuery
    from polylogue.archive.message.roles import Role
    from polylogue.archive.session.branch_type import BranchType
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession

    def prose(native_id: str, role: Role, texts: list[str]) -> ParsedMessage:
        return ParsedMessage(
            provider_message_id=native_id,
            role=role,
            blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text) for text in texts],
        )

    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="context-parent",
        messages=[prose("p0", Role.USER, ["first\nsecond", "third"]), prose("p1", Role.ASSISTANT, ["parent reply"])],
    )
    child = action_stream("context-child", [("use", "use", "tool", None), ("result", "result", "tool", False)])
    child.parent_session_provider_id = "context-parent"
    child.branch_type = BranchType.FORK
    child.messages = [
        prose("c0", Role.USER, ["first\nsecond", "third"]),
        prose("c1", Role.ASSISTANT, ["parent reply"]),
        child.messages[0],
        prose("during", Role.ASSISTANT, ["while the tool ran"]),
        child.messages[1],
        prose("after", Role.USER, ["x" * 2000, "next\nline"]),
        prose("last", Role.ASSISTANT, ["last"]),
    ]
    with ArchiveStore(tmp_path / "archive") as archive:
        parent_id = write_index_session(archive, parent)
        child_id = write_index_session(archive, child)
        assert (
            archive._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (child_id,)).fetchone()[0] == 5
        )
        edge = archive._conn.execute(
            "SELECT inheritance, resolved_dst_session_id FROM session_links WHERE src_session_id = ?", (child_id,)
        ).fetchone()
        assert tuple(edge) == ("prefix-sharing", parent_id)
        episode = archive.list_tool_episode_insights(ToolEpisodeQuery(session_id=child_id))[0]
        assert episode.context_before == ("user: first\nsecond\nthird", "assistant: parent reply")
        assert episode.context_after == ("user: " + "x" * 2000 + "\nnext\nline", "assistant: last")
        assert episode.next_action == "x" * 2000 + "\nnext\nline"
        assert episode.result_output == "result"
        assert episode.result_state == "outcome_success"
        assert not archive._conn.in_transaction


def test_observed_event_evidence_subqueries_read_materialized_results(tmp_path: Path) -> None:
    """Per-use evidence lookups must not re-derive the association windows.

    Inlining ``association_results`` into the correlated evidence and search
    subqueries re-runs the windowed association once per paired use, which is
    quadratic in a session's tool results: a Codex archive with tens of
    thousands of tool calls held the status and health readers for minutes.
    """
    events = [
        event
        for index in range(40)
        for event in ((f"u{index}", "use", f"t{index}", None), (f"r{index}", "result", f"t{index}", False))
    ]
    with ArchiveStore(tmp_path / "archive") as archive:
        write_index_session(archive, action_stream("plan", events))
        plan = archive._conn.execute(
            "EXPLAIN QUERY PLAN " + observed_event_relation_sql(source_where="1") + " SELECT * FROM observed_events"
        ).fetchall()
    children: dict[int, list[tuple[int, str]]] = {}
    for node, parent, _unused, detail in plan:
        children.setdefault(parent, []).append((node, detail))

    def subtree(node: int) -> list[str]:
        # A MATERIALIZE child runs once per statement, not per correlated row.
        return [
            detail
            for child, detail in children.get(node, [])
            if not detail.startswith("MATERIALIZE")
            for detail in (detail, *subtree(child))
        ]

    correlated = [node for node, _parent, _unused, detail in plan if detail.startswith("CORRELATED")]
    assert len(correlated) == 2
    for node in correlated:
        details = subtree(node)
        assert not [detail for detail in details if "association_numbered" in detail], details
        assert [detail for detail in details if detail.startswith("SEARCH tr ")], details
