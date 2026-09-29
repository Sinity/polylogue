"""Association evidence is causal, ambiguity-visible and shared by every reader."""

from __future__ import annotations

from pathlib import Path

import pytest

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
    """Ordinal joins fail orphan/gap/duplicate cases; nearest joins fail parallel uses."""
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
        columns = "tool_command, tool_result_block_id, tool_outcome, result_state, outcome_unknown_reason"
        materialized_sql = f"SELECT {columns} FROM actions WHERE session_id = ? ORDER BY tool_command"
        assert [tuple(row) for row in conn.execute(materialized_sql, (session_id,))] == expected_rows
        for bounded in (False, True):
            sql = action_relation_select_sql(session_placeholders="?" if bounded else None)
            parameters = (session_id,) * 3 if bounded else ()
            rows = conn.execute(f"SELECT {columns} FROM ({sql}) ORDER BY tool_command", parameters).fetchall()
            assert [tuple(row) for row in rows] == expected_rows
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
        assert [tuple(row) for row in conn.execute(materialized_sql, (session_id,))] == expected_rows
        rebuild_all_action_pairs_sync(conn)
        assert [tuple(row) for row in conn.execute(materialized_sql, (session_id,))] == expected_rows
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        conn.commit()
    with ArchiveStore.open_existing(tmp_path / "archive") as reopened:
        assert [tuple(row) for row in reopened._conn.execute(materialized_sql, (session_id,))] == expected_rows


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
