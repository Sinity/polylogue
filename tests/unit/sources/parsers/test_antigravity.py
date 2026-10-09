from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider, TitleSource
from polylogue.pipeline.ids import session_revision_projection
from polylogue.sources.parsers.antigravity import (
    AntigravitySessionSummary,
    _mark_active_leaf,
    looks_like_trajectory_db_path,
    parse_markdown_export,
)
from polylogue.sources.parsers.base import ParsedMessage
from polylogue.sources.streamed_event_payload import StreamedJsonArray
from tests.infra.antigravity_parser import parse_trajectory_db


def test_parse_markdown_export_splits_known_sections() -> None:
    markdown = """# Chat Session

Note: _This is purely the output of the chat session._

### User Input

Run pytest.

### Planner Response

The focused checks passed.
"""
    summary = AntigravitySessionSummary(
        cascade_id="cascade-1",
        title="Focused checks",
        workspace_name="polylogue",
        snippet="Run pytest.",
        last_modified_time="2026-03-05T04:21:34Z",
    )

    session = parse_markdown_export(markdown, summary)

    assert session.source_name is Provider.ANTIGRAVITY
    assert session.provider_session_id == "cascade-1"
    assert session.title == "Focused checks"
    assert session.title_source is TitleSource.ORIGIN
    assert session.updated_at == "2026-03-05T04:21:34Z"
    assert [message.role for message in session.messages] == [Role.USER, Role.ASSISTANT]
    assert [message.text for message in session.messages] == [
        "Run pytest.",
        "The focused checks passed.",
    ]
    assert session.messages[0].blocks[0].type == BlockType.TEXT
    assert session.messages[0].provider_message_id.startswith("synthetic-")
    assert session.messages[1].provider_message_id.startswith("synthetic-")
    assert session.messages[0].provider_message_id != session.messages[1].provider_message_id
    assert [message.position for message in session.messages] == [0, 1]
    assert [message.variant_index for message in session.messages] == [0, 0]
    assert [message.is_active_path for message in session.messages] == [True, True]
    assert [message.is_active_leaf for message in session.messages] == [False, True]
    assert session.active_leaf_message_provider_id == session.messages[1].provider_message_id


def test_parse_markdown_export_reordering_keeps_synthetic_revision_identity() -> None:
    summary = AntigravitySessionSummary(cascade_id="cascade-order")
    forward = parse_markdown_export(
        "### User Input\n\nQuestion\n\n### Planner Response\n\nAnswer\n",
        summary,
    )
    reordered = parse_markdown_export(
        "### Planner Response\n\nAnswer\n\n### User Input\n\nQuestion\n",
        summary,
    )

    assert (
        session_revision_projection(forward).message_contents == session_revision_projection(reordered).message_contents
    )


def test_mark_active_leaf_flags_exactly_one_message_with_duplicate_ids() -> None:
    """bd polylogue-2hwl: a duplicate ``provider_message_id`` (a retried or
    regenerated section reusing the same id) must not produce more than one
    ``is_active_leaf=True`` message -- the pre-fix comparison
    (``message.provider_message_id == active_leaf_message_provider_id``)
    flagged every position sharing that id, not just the true final one.
    """
    messages = [
        ParsedMessage(provider_message_id="dup", role=Role.USER, text="first", position=0, is_active_path=True),
        ParsedMessage(provider_message_id="other", role=Role.ASSISTANT, text="middle", position=1, is_active_path=True),
        ParsedMessage(provider_message_id="dup", role=Role.ASSISTANT, text="final", position=2, is_active_path=True),
    ]

    marked = _mark_active_leaf(messages)

    leaves = [message for message in marked if message.is_active_leaf]
    assert len(leaves) == 1
    assert leaves[0].text == "final"


def test_markdown_activity_ids_survive_unrelated_insertion_and_keep_identical_occurrences(tmp_path: Path) -> None:
    """Putting a global activity ordinal in the semantic seed renames both edits."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.live_ingest import write_index_session

    body = "A\n\n*Edited relevant file*\n\nB\n\n*Edited relevant file*\n"
    summary = AntigravitySessionSummary(cascade_id="activity-identity")
    before = parse_markdown_export("### Planner Response\n\n" + body, summary)
    after = parse_markdown_export("### Planner Response\n\n*Checked command status*\n\nInserted\n\n" + body, summary)
    activities = [message for message in before.messages if message.blocks[0].type is BlockType.TOOL_USE]
    assert len(activities) == 2
    assert all(message.provider_message_id == "" for message in activities)
    tool_name = activities[0].blocks[0].tool_name
    with ArchiveStore(tmp_path / "archive") as archive:
        session_id = write_index_session(archive, before)

        def edit_ids() -> list[str]:
            return [
                row[0]
                for row in archive.index_connection.execute(
                    "SELECT m.message_id FROM messages m JOIN blocks b ON b.message_id=m.message_id "
                    "WHERE m.session_id=? AND b.tool_name=? ORDER BY m.position",
                    (session_id, tool_name),
                )
            ]

        original = edit_ids()
        write_index_session(archive, after)
        assert edit_ids() == original
        assert len(set(original)) == 2


def test_parse_markdown_export_falls_back_to_single_export_message() -> None:
    markdown = """# Chat Session

Note: generated export

Unstructured transcript body.
"""
    summary = AntigravitySessionSummary(cascade_id="cascade-2")

    session = parse_markdown_export(markdown, summary)

    assert [message.role for message in session.messages] == [Role.ASSISTANT]
    assert session.messages[0].provider_message_id.startswith("synthetic-")
    assert session.messages[0].text == "Unstructured transcript body."
    assert session.messages[0].position == 0
    assert session.messages[0].is_active_leaf is True
    assert session.active_leaf_message_provider_id == session.messages[0].provider_message_id
    assert session.title_source is None


def test_parse_markdown_export_has_no_degraded_flag() -> None:
    """Language-server export sessions are whole transcripts — not fragmented.

    Markdown export sessions must NOT carry the brain-metadata fragment flag;
    they represent complete work sessions and must not be excluded from primary
    views by the same filter that suppresses brain-metadata fragments.
    """
    markdown = "### User Input\n\nRun checks.\n\n### Planner Response\n\nDone.\n"
    summary = AntigravitySessionSummary(
        cascade_id="cascade-clean",
        title="Clean session",
        last_modified_time="2026-04-01T12:00:00Z",
    )

    session = parse_markdown_export(markdown, summary)

    assert session.ingest_flags == []


def _trajectory_db(path: Path, *, malformed: bool = False) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
        CREATE TABLE steps (
            idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT,
            status TEXT, error_details TEXT
        );
        CREATE TABLE conversation_summaries (cascade_id TEXT, title TEXT, last_modified_time TEXT);
        CREATE TABLE parent_references (cascade_id TEXT, parent_id TEXT);
        """
    )
    connection.execute("INSERT INTO trajectory_meta VALUES (?, ?)", ("trajectory-1", "cascade-1"))
    connection.execute(
        "INSERT INTO conversation_summaries VALUES (?, ?, ?)",
        ("cascade-1", "Trajectory title", "2026-03-05T04:21:34Z"),
    )
    connection.execute("INSERT INTO parent_references VALUES (?, ?)", ("cascade-1", "parent-1"))
    connection.executemany(
        "INSERT INTO steps VALUES (?, ?, ?, ?, ?, ?)",
        [
            (0, "message", "v1", '{"role":"user","text":"hello"}', None, None),
            (1, "terminal_command", "v1", '{"tool_name":"shell","command":"printf hi"}', None, None),
            (2, "tool_result", "v1", '{"tool_name":"shell","output":"hi","status":"success"}', None, None),
            (3, "file_edit", "v1", '{"path":"README.md","old_string":"old","new_string":"new"}', None, None),
            (4, "future_step", "future", '{"opaque":true}' if not malformed else "not-json", None, None),
        ],
    )
    connection.commit()
    connection.close()
    return path


def test_trajectory_sqlite_parser_preserves_identity_order_tools_and_summary(tmp_path: Path) -> None:
    path = _trajectory_db(tmp_path / "conversation.db")

    assert looks_like_trajectory_db_path(path)
    sessions = list(parse_trajectory_db(path))

    assert len(sessions) == 1
    session = sessions[0]
    assert session.provider_session_id == "trajectory-1"
    assert session.provider_session_aliases == ["cascade-1"]
    assert session.title == "Trajectory title"
    assert [message.position for message in session.messages] == [0, 1, 2, 3]
    assert session.messages[1].blocks[0].tool_name == "shell"
    assert session.messages[1].blocks[0].tool_input == {"command": "printf hi"}
    result = session.messages[2].blocks[0]
    assert result.text == "hi"
    assert result.is_error is False
    assert session.messages[3].blocks[0].file_edit is not None
    assert any(event.event_type == "antigravity_parent_reference" for event in session.session_events)
    assert any(event.event_type == "antigravity_unmatched_parent_reference" for event in session.session_events)
    assert session.parent_session_provider_id == "parent-1"


def test_trajectory_sqlite_parser_refuses_known_text_with_unknown_step_format(tmp_path: Path) -> None:
    path = _trajectory_db(tmp_path / "conversation.db")
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE steps SET step_format = 'future-v9' WHERE idx = 0")

    session = list(parse_trajectory_db(path))[0]

    assert [message.text for message in session.messages] == [None, "hi", None]
    unsupported = [event for event in session.session_events if event.event_type == "antigravity_unsupported_step"]
    assert unsupported
    assert unsupported[0].payload["reason"] == "unsupported_step_format_or_type"
    assert "degraded:unsupported-trajectory-steps" in session.ingest_flags


def test_trajectory_sqlite_parser_empty_schema_is_attributable(tmp_path: Path) -> None:
    path = tmp_path / "empty.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
            CREATE TABLE steps (idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT);
            """
        )

    sessions = list(parse_trajectory_db(path, fallback_id="empty"))

    assert len(sessions) == 1
    assert sessions[0].messages == []
    assert any(event.event_type == "antigravity_trajectory_empty" for event in sessions[0].session_events)


def test_trajectory_sqlite_parser_does_not_merge_unkeyed_multiple_trajectories(tmp_path: Path) -> None:
    path = _trajectory_db(tmp_path / "conversation.db")
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO trajectory_meta VALUES (?, ?)", ("trajectory-2", "cascade-2"))

    sessions = list(parse_trajectory_db(path))

    assert [session.provider_session_id for session in sessions] == ["trajectory-1", "trajectory-2"]
    assert all(session.messages == [] for session in sessions)
    assert all(
        any(event.event_type == "antigravity_unattributed_steps" for event in session.session_events)
        for session in sessions
    )


def test_trajectory_sqlite_parser_retains_unmatched_summary_as_metadata(tmp_path: Path) -> None:
    path = _trajectory_db(tmp_path / "conversation.db")
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO conversation_summaries VALUES (?, ?, ?)",
            ("orphan-cascade", "Orphan title", "2026-03-06T04:21:34Z"),
        )

    sessions = list(parse_trajectory_db(path))

    orphan = next(session for session in sessions if session.provider_session_id == "orphan-cascade")
    assert orphan.messages == []
    assert orphan.ingest_flags == ["degraded:unmatched-trajectory-summary"]
    event = next(event for event in orphan.session_events if event.event_type == "antigravity_unmatched_summary")
    assert event.payload["title"] == "Orphan title"


def test_trajectory_sqlite_parser_reserves_summary_keys_against_row_fallback_ids(tmp_path: Path) -> None:
    """A row-fallback id must not collide with an unmatched summary's key.

    Anti-vacuity (Codex P2, #5711): an anonymous ``trajectory_meta`` row (no
    ``trajectory_id``/``cascade_id``) takes the path-derived fallback; an
    unrelated ``conversation_summaries`` row keyed by that exact string is
    yielded as its own session under the unmatched-summary branch. Without
    reserving summary keys, both land under one ``provider_session_id``.
    """
    path = tmp_path / "conversation.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
            CREATE TABLE steps (idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT);
            CREATE TABLE conversation_summaries (cascade_id TEXT, title TEXT, last_modified_time TEXT);
            """
        )
        connection.execute("INSERT INTO trajectory_meta VALUES (NULL, NULL)")
        connection.execute(
            "INSERT INTO conversation_summaries VALUES (?, ?, ?)", ("x", "Unrelated summary", "2026-03-06T04:21:34Z")
        )

    sessions = list(parse_trajectory_db(path, fallback_id="x"))

    provider_ids = [session.provider_session_id for session in sessions]
    assert len(provider_ids) == len(set(provider_ids)) == 2, f"colliding provider_session_id: {provider_ids}"
    orphan = next(session for session in sessions if session.ingest_flags == ["degraded:unmatched-trajectory-summary"])
    assert orphan.provider_session_id == "x"


def test_several_anonymous_trajectories_are_refused(tmp_path: Path) -> None:
    """Unidentified trajectories with nothing stable between them are not guessed apart.

    Anti-vacuity (Codex P1, #5711): key each on its implicit rowid and a
    ``VACUUM`` after deleting the first renumbers the survivor onto the
    deleted trajectory's identity.
    """
    from polylogue.sources.sqlite_export import LogicalExportError

    path = tmp_path / "conversation.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
            CREATE TABLE steps (idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT);
            INSERT INTO trajectory_meta VALUES (NULL, NULL);
            INSERT INTO trajectory_meta VALUES (NULL, NULL);
            """
        )

    with pytest.raises(LogicalExportError, match="no stable identity"):
        list(parse_trajectory_db(path, fallback_id="x"))


def test_the_bare_fallback_is_not_taken_when_a_native_id_occupies_it(tmp_path: Path) -> None:
    """An anonymous first row never takes a fallback another row already names.

    Anti-vacuity (Codex P2, #5711): keep the bare fallback for the first
    anonymous row unconditionally and it shares ``x`` with the row whose
    native trajectory id is ``x``.
    """
    path = tmp_path / "conversation.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
            CREATE TABLE steps (idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT);
            INSERT INTO trajectory_meta VALUES (NULL, NULL);
            INSERT INTO trajectory_meta VALUES ('x', NULL);
            """
        )
    ids = [session.provider_session_id for session in parse_trajectory_db(path, fallback_id="x")]

    assert len(ids) == 2
    assert len(set(ids)) == 2


def test_trajectory_sqlite_parser_refuses_malformed_step_without_fabricating_text(tmp_path: Path) -> None:
    path = _trajectory_db(tmp_path / "conversation.db", malformed=True)

    session = list(parse_trajectory_db(path))[0]

    assert len(session.messages) == 4
    assert any(event.event_type == "antigravity_unsupported_step" for event in session.session_events)
    assert "degraded:unsupported-trajectory-steps" in session.ingest_flags


def test_trajectory_step_identity_uses_the_source_index(tmp_path: Path) -> None:
    """A later step's identity must not depend on an earlier step's admission.

    Anti-vacuity: key the fallback on the count of materialized messages and
    the unchanged step at ``idx=1`` is renamed from ``:step:1`` to ``:step:0``
    the moment the malformed step at ``idx=0`` becomes unparseable.
    """
    path = _trajectory_db(tmp_path / "conversation.db")
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM steps WHERE idx > 1")
        connection.execute("UPDATE steps SET step_payload = 'not-json' WHERE idx = 0")

    session = list(parse_trajectory_db(path))[0]

    assert [message.provider_message_id for message in session.messages] == ["trajectory:step:1"]


def test_trajectory_user_step_is_classified_human_authored(tmp_path: Path) -> None:
    """An explicit ``role`` is positive material-origin evidence.

    Anti-vacuity: drop the classification and the prompt stays ``UNKNOWN``,
    which excludes it from human-authored/user-word and cost accounting.
    """
    path = _trajectory_db(tmp_path / "conversation.db")

    session = list(parse_trajectory_db(path))[0]

    user_message = session.messages[0]
    assert user_message.role is Role.USER
    assert user_message.material_origin is MaterialOrigin.HUMAN_AUTHORED
    # The opposite direction: a tool result is not human-authored.
    assert session.messages[2].material_origin is not MaterialOrigin.HUMAN_AUTHORED


def _keyed_trajectory_db(path: Path, *, key_value: str | None) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
        CREATE TABLE steps (
            idx INTEGER, trajectory_id TEXT, cascade_id TEXT, step_type TEXT,
            step_format TEXT, step_payload TEXT, status TEXT, error_details TEXT
        );
        CREATE TABLE conversation_summaries (cascade_id TEXT, title TEXT, last_modified_time TEXT);
        """
    )
    connection.execute("INSERT INTO trajectory_meta VALUES (?, ?)", ("trajectory-1", "cascade-1"))
    connection.execute(
        "INSERT INTO steps VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (0, key_value, key_value, "message", "v1", '{"role":"user","text":"hello"}', None, None),
    )
    connection.commit()
    connection.close()
    return path


def test_sole_trajectory_with_null_step_keys_keeps_its_steps(tmp_path: Path) -> None:
    """A declared-but-unset key column must not erase the only trajectory.

    Anti-vacuity: leave the keyed query as the only branch and this session
    reports ``step_count: 0`` with an accounting denominator of 0 -- every
    real step silently absent from both messages and admission.
    """
    path = _keyed_trajectory_db(tmp_path / "legacy.db", key_value=None)

    session = list(parse_trajectory_db(path))[0]

    assert [message.text for message in session.messages] == ["hello"]
    assert not any(event.event_type == "antigravity_trajectory_empty" for event in session.session_events)


def test_keyed_steps_are_not_reattributed_to_a_foreign_trajectory(tmp_path: Path) -> None:
    """The opposite direction: a real key that does not match stays unmatched."""
    path = _keyed_trajectory_db(tmp_path / "keyed.db", key_value="other-trajectory")

    session = list(parse_trajectory_db(path))[0]

    assert session.messages == []


def test_conflicting_parent_references_assert_no_parent(tmp_path: Path) -> None:
    """Disagreeing parent rows are an ambiguity, not a first-row choice.

    Anti-vacuity: take ``matching_parent_refs[0]`` and whichever row the
    unordered SELECT returns first becomes a durable topology edge.
    """
    path = _trajectory_db(tmp_path / "conversation.db")
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO parent_references VALUES (?, ?)", ("cascade-1", "parent-2"))

    session = list(parse_trajectory_db(path))[0]

    assert session.parent_session_provider_id is None
    ambiguity = next(
        event for event in session.session_events if event.event_type == "antigravity_ambiguous_parent_reference"
    )
    parent_provider_ids = ambiguity.payload["parent_provider_ids"]
    assert isinstance(parent_provider_ids, StreamedJsonArray)
    assert list(parent_provider_ids.iter_values()) == ["parent-1", "parent-2"]
    parent_event = next(event for event in session.session_events if event.event_type == "antigravity_parent_reference")
    references = parent_event.payload["references"]
    assert isinstance(references, StreamedJsonArray)
    assert list(references.iter_values()) == [
        {"cascade_id": "cascade-1", "parent_id": "parent-1"},
        {"cascade_id": "cascade-1", "parent_id": "parent-2"},
    ]


def test_single_parent_reference_still_asserts_its_edge(tmp_path: Path) -> None:
    """The opposite direction: a blanket refusal must not pass."""
    path = _trajectory_db(tmp_path / "conversation.db")

    session = list(parse_trajectory_db(path))[0]

    assert session.parent_session_provider_id == "parent-1"


def _steps_db(path: Path, steps: list[tuple[int, str, str]]) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
        CREATE TABLE steps (idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT);
        """
    )
    connection.execute("INSERT INTO trajectory_meta VALUES ('trajectory-1', 'cascade-1')")
    connection.executemany(
        "INSERT INTO steps VALUES (?, ?, 'v1', ?)", [(idx, step_type, payload) for idx, step_type, payload in steps]
    )
    connection.commit()
    connection.close()
    return path


def test_trajectory_id_less_tool_steps_get_step_keyed_ids_and_pair_their_result(tmp_path: Path) -> None:
    """An ID-less call is keyed by its own step; its adjacent result reports it.

    Anti-vacuity: drop the synthesized id and the call and file edit carry
    ``tool_id=None`` again, so the result cannot pair and no file edit can
    be keyed.
    """
    path = _trajectory_db(tmp_path / "conversation.db")

    [first] = list(parse_trajectory_db(path))
    [again] = list(parse_trajectory_db(path))

    call, result, edit = (first.messages[index].blocks[0] for index in (1, 2, 3))
    assert call.type is BlockType.TOOL_USE
    assert call.tool_id == "trajectory:step:1:tool"
    assert result.type is BlockType.TOOL_RESULT
    assert result.tool_id == call.tool_id
    assert edit.type is BlockType.TOOL_USE
    assert edit.tool_id == "trajectory:step:3:tool"
    assert edit.file_edit is not None
    assert [message.blocks[0].tool_id for message in again.messages] == [
        message.blocks[0].tool_id for message in first.messages
    ]


def test_trajectory_id_less_tool_steps_persist_their_pairing_and_file_edit(tmp_path: Path) -> None:
    """The archive writer pairs the synthesized ids and keys the file edit."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.live_ingest import write_index_session

    [session] = list(parse_trajectory_db(_trajectory_db(tmp_path / "conversation.db")))

    with ArchiveStore(tmp_path / "archive") as archive:
        session_id = write_index_session(archive, session)
        index_path = archive.index_db_path
    with sqlite3.connect(index_path) as conn:
        uses = conn.execute(
            "SELECT tool_id, tool_outcome, block_id FROM blocks "
            "WHERE session_id = ? AND block_type = 'tool_use' ORDER BY tool_id",
            (session_id,),
        ).fetchall()
        edits = conn.execute(
            "SELECT tool_use_block_id, file_path, old_string, new_string FROM file_edits WHERE session_id = ?",
            (session_id,),
        ).fetchall()

    assert [(tool_id, outcome) for tool_id, outcome, _block in uses] == [
        ("trajectory:step:1:tool", "ok"),
        ("trajectory:step:3:tool", "no_result"),
    ]
    edit_block_id = uses[1][2]
    assert edits == [(edit_block_id, "README.md", "old", "new")]


@pytest.mark.parametrize(
    ("steps", "paired"),
    [
        # Two calls in flight: nothing says which one the result reports.
        (
            [
                (0, "terminal_command", '{"tool_name":"shell","command":"a"}'),
                (1, "terminal_command", '{"tool_name":"shell","command":"b"}'),
                (2, "tool_result", '{"tool_name":"shell","output":"x","status":"success"}'),
            ],
            False,
        ),
        # A step between the call and the result breaks the adjacency.
        (
            [
                (0, "terminal_command", '{"tool_name":"shell","command":"a"}'),
                (1, "message", '{"role":"assistant","text":"waiting"}'),
                (2, "tool_result", '{"tool_name":"shell","output":"x","status":"success"}'),
            ],
            False,
        ),
        # A declared tool name that differs refutes the pairing.
        (
            [
                (0, "terminal_command", '{"tool_name":"shell","command":"a"}'),
                (1, "tool_result", '{"tool_name":"browser","output":"x","status":"success"}'),
            ],
            False,
        ),
        # An id-bearing call is answered by its own id, never by adjacency.
        (
            [
                (0, "terminal_command", '{"tool_name":"shell","command":"a","call_id":"c-1"}'),
                (1, "tool_result", '{"tool_name":"shell","output":"x","status":"success"}'),
            ],
            False,
        ),
        (
            [
                (0, "terminal_command", '{"tool_name":"shell","command":"a"}'),
                (1, "tool_result", '{"output":"x","status":"success"}'),
            ],
            True,
        ),
    ],
    ids=["two-in-flight", "interleaved-step", "name-mismatch", "id-bearing-call", "adjacent"],
)
def test_trajectory_id_less_result_pairs_only_an_unambiguous_adjacent_call(
    tmp_path: Path, steps: list[tuple[int, str, str]], paired: bool
) -> None:
    [session] = list(parse_trajectory_db(_steps_db(tmp_path / "conversation.db", steps)))

    [result] = [
        block for message in session.messages for block in message.blocks if block.type is BlockType.TOOL_RESULT
    ]
    call_ids = {
        block.tool_id for message in session.messages for block in message.blocks if block.type is BlockType.TOOL_USE
    }
    if paired:
        assert result.tool_id in call_ids
    else:
        assert result.tool_id is None
