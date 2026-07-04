from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import (
    BlockType,
    MaterialOrigin,
    MessageType,
    Provider,
    SessionKind,
    TitleSource,
    WebConstructType,
)
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedPasteEvidence,
    ParsedSession,
    ParsedSessionEvent,
    ParsedWebConstruct,
)
from polylogue.storage.sqlite.archive_tiers import write as archive_tier_write
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    ArchiveAgentPolicy,
    ArchiveInsightMaterialization,
    ArchiveSessionPhase,
    ArchiveSessionTag,
    ArchiveSessionWorkEvent,
    read_archive_session_envelope,
    read_insight_materialization,
    read_session_agent_policies,
    read_session_phases,
    read_session_tags,
    read_session_work_events,
    search_archive_blocks,
    upsert_insight_materialization,
    upsert_session_phase,
    upsert_session_profile_costs,
    upsert_session_tag,
    upsert_session_work_event,
    write_parsed_session_to_archive,
)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def test_message_content_hash_tracks_same_identity_body_edits(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    try:
        first = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="same-id-edit",
            title="Same id edit",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.USER,
                    text="first body long enough for embeddings",
                    material_origin=MaterialOrigin.HUMAN_AUTHORED,
                )
            ],
        )
        session_id = write_parsed_session_to_archive(conn, first)
        first_hash = conn.execute(
            "SELECT content_hash FROM messages WHERE session_id = ? AND native_id = 'm1'",
            (session_id,),
        ).fetchone()[0]

        second = first.model_copy(
            update={
                "messages": [
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="edited body long enough for embeddings",
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    )
                ]
            }
        )
        write_parsed_session_to_archive(conn, second)
        second_hash = conn.execute(
            "SELECT content_hash FROM messages WHERE session_id = ? AND native_id = 'm1'",
            (session_id,),
        ).fetchone()[0]

        assert first_hash != second_hash
    finally:
        conn.close()


def test_archive_tiers_writer_materializes_typed_web_constructs(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="construct-session",
        title="Construct session",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                text="answer with source",
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TEXT,
                        text="answer with source",
                        web_constructs=[
                            ParsedWebConstruct(
                                construct_type=WebConstructType.CONTENT_REFERENCE,
                                provider_key="content_references",
                                title="Source",
                                url="https://example.test/source",
                                rank=0,
                            ),
                            ParsedWebConstruct(
                                construct_type=WebConstructType.CANVAS,
                                provider_key="canvas",
                                source_id="canvas-1",
                            ),
                        ],
                    )
                ],
            )
        ],
    )
    session_id = write_parsed_session_to_archive(conn, session)

    rows = conn.execute(
        """
        SELECT construct_type, provider_key, title, url, source_id, rank
        FROM web_content_constructs
        WHERE session_id = ?
        ORDER BY position
        """,
        (session_id,),
    ).fetchall()
    assert [row["construct_type"] for row in rows] == ["content_reference", "canvas"]
    assert rows[0]["provider_key"] == "content_references"
    assert rows[0]["title"] == "Source"
    assert rows[0]["url"] == "https://example.test/source"
    assert rows[0]["rank"] == 0
    assert rows[1]["source_id"] == "canvas-1"

    envelope = read_archive_session_envelope(conn, session_id)
    assert envelope is not None
    assert envelope.messages[0].blocks[0].metadata is None


def test_archive_tiers_writer_materializes_codex_session(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-session-1",
        title="Writer evidence",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:02+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="u1",
                role=Role.USER,
                text="run focused checks",
                material_origin=MaterialOrigin.HUMAN_AUTHORED,
                timestamp="2026-01-01T00:00:01+00:00",
                position=0,
                variant_index=0,
                is_active_path=True,
                is_active_leaf=False,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="run focused checks")],
            ),
            ParsedMessage(
                provider_message_id="a1",
                role=Role.ASSISTANT,
                text="checks passed",
                timestamp="2026-01-01T00:00:02+00:00",
                position=1,
                variant_index=0,
                is_active_path=True,
                is_active_leaf=True,
                model_name="gpt-5-codex",
                model_effort="high",
                duration_ms=1200,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="exec_command",
                        tool_id="tool-1",
                        tool_input={"command": "pytest -q", "path": "tests"},
                    ),
                    ParsedContentBlock(
                        type=BlockType.TOOL_RESULT,
                        tool_id="tool-1",
                        text="checks passed",
                        is_error=False,
                        exit_code=0,
                    ),
                ],
            ),
        ],
        active_leaf_message_provider_id="a1",
    )

    session_id = write_parsed_session_to_archive(conn, session)
    envelope = read_archive_session_envelope(conn, session_id)

    assert envelope.session_id == "codex-session:codex-session-1"
    assert envelope.origin == "codex-session"
    assert envelope.active_leaf_message_id == "codex-session:codex-session-1:a1"
    assert [message.message_id for message in envelope.messages] == [
        "codex-session:codex-session-1:u1",
        "codex-session:codex-session-1:a1",
    ]
    assert envelope.messages[1].is_active_leaf is True
    assert [block.block_type for block in envelope.messages[1].blocks] == ["tool_use", "tool_result"]
    assert envelope.messages[1].blocks[1].tool_result_is_error == 0
    assert envelope.messages[1].blocks[1].tool_result_exit_code == 0
    assert [message.material_origin for message in envelope.messages] == ["human_authored", "assistant_authored"]
    action = conn.execute("SELECT tool_command, output_text, is_error, exit_code FROM actions").fetchone()
    assert dict(action) == {
        "tool_command": "pytest -q",
        "output_text": "checks passed",
        "is_error": 0,
        "exit_code": 0,
    }
    assert search_archive_blocks(conn, "focused") == ["codex-session:codex-session-1:u1:0"]


def test_archive_tiers_writer_splits_provider_user_from_authored_user_counts(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="claude-code-authoredness",
        title="authoredness",
        messages=[
            ParsedMessage(
                provider_message_id="runtime-protocol",
                role=Role.USER,
                text="<task-notification>done</task-notification>",
                message_type=MessageType.PROTOCOL,
                material_origin=MaterialOrigin.RUNTIME_PROTOCOL,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="<task-notification>done</task-notification>")],
            ),
            ParsedMessage(
                provider_message_id="generated-pack",
                role=Role.USER,
                text="# Commit N: Generate all artifacts\n\nnot typed by the operator",
                material_origin=MaterialOrigin.GENERATED_CONTEXT_PACK,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TEXT,
                        text="# Commit N: Generate all artifacts\n\nnot typed by the operator",
                    )
                ],
            ),
            ParsedMessage(
                provider_message_id="typed-prompt",
                role=Role.USER,
                text="Please inspect the failing parser.",
                material_origin=MaterialOrigin.HUMAN_AUTHORED,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="Please inspect the failing parser.")],
            ),
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)
    counts = conn.execute(
        """
        SELECT user_message_count, authored_user_message_count,
               user_word_count, authored_user_word_count
        FROM sessions
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()

    assert dict(counts) == {
        "user_message_count": 3,
        "authored_user_message_count": 1,
        "user_word_count": 17,
        "authored_user_word_count": 5,
    }


def test_archive_store_connection_applies_canonical_profile(tmp_path: Path) -> None:
    """ArchiveStore must apply the write/read connection profile, not a bare connect.

    A bare sqlite3.connect defaults to a 5s busy_timeout with no WAL tuning.
    Under daemon write contention (live ingest vs convergence both writing
    index.db) that window is exceeded and ingest fails with "database is
    locked". The write profile raises busy_timeout to 30s so writers queue.
    """
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore(tmp_path, initialize=True, read_only=False) as archive:
        assert int(archive._conn.execute("PRAGMA busy_timeout").fetchone()[0]) == 30_000
        assert str(archive._conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"

    with ArchiveStore(tmp_path, initialize=False, read_only=True) as archive:
        assert int(archive._conn.execute("PRAGMA busy_timeout").fetchone()[0]) == 5_000
        assert int(archive._conn.execute("PRAGMA query_only").fetchone()[0]) == 1

    with ArchiveStore(tmp_path, initialize=False, read_only=True, read_timeout=1.0) as archive:
        assert int(archive._conn.execute("PRAGMA busy_timeout").fetchone()[0]) == 1_000
        assert int(archive._conn.execute("PRAGMA query_only").fetchone()[0]) == 1


def test_archive_tiers_writer_ingests_session_with_root_cwd_and_no_repo_name(tmp_path: Path) -> None:
    """A session whose only working directory is "/" must still ingest.

    ``_repo_name`` returns None when no name can be derived from the cwd (e.g.
    "/" or "."), but ``repos.repo_name`` is ``NOT NULL``. Passing the None
    through crashed the write with an IntegrityError, silently dropping the
    session and retrying it forever. The writer must persist the empty-string
    sentinel instead and keep the session.
    """
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="root-cwd-1",
        title="Session started at filesystem root",
        working_directories=["/"],
        messages=[
            ParsedMessage(
                provider_message_id="u1",
                role=Role.USER,
                text="hello from root",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hello from root")],
            ),
        ],
    )

    # Must not raise sqlite3.IntegrityError on repos.repo_name NOT NULL.
    session_id = write_parsed_session_to_archive(conn, session)

    repos = [dict(row) for row in conn.execute("SELECT root_path, repo_name FROM repos").fetchall()]
    assert {"root_path": "/", "repo_name": ""} in repos
    envelope = read_archive_session_envelope(conn, session_id)
    assert len(envelope.messages) == 1


def test_archive_tiers_writer_does_not_collapse_duplicate_message_native_ids(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-duplicate-native-ids",
        title="Duplicate native IDs",
        messages=[
            ParsedMessage(
                provider_message_id="dup",
                role=Role.USER,
                text="first",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="first")],
            ),
            ParsedMessage(
                provider_message_id="dup",
                role=Role.ASSISTANT,
                text="second",
                position=1,
                is_active_leaf=True,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="second")],
            ),
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    message_rows = conn.execute(
        """
        SELECT message_id, native_id, position, role
        FROM messages
        WHERE session_id = ?
        ORDER BY position
        """,
        (session_id,),
    ).fetchall()
    block_rows = conn.execute(
        """
        SELECT message_id, text
        FROM blocks
        WHERE session_id = ?
        ORDER BY message_id
        """,
        (session_id,),
    ).fetchall()
    session_row = conn.execute(
        "SELECT message_count, active_leaf_message_id FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()

    assert [(row["position"], row["role"], row["native_id"]) for row in message_rows] == [
        (0, "user", None),
        (1, "assistant", None),
    ]
    assert [row["message_id"] for row in message_rows] == [f"{session_id}:0.0", f"{session_id}:1.0"]
    assert [(row["message_id"], row["text"]) for row in block_rows] == [
        (f"{session_id}:0.0", "first"),
        (f"{session_id}:1.0", "second"),
    ]
    assert session_row["message_count"] == 2
    assert session_row["active_leaf_message_id"] == f"{session_id}:1.0"


def test_archive_tiers_writer_replaces_lone_surrogates_before_sqlite(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="surrogate-session",
        title="Surrogate \udce2 title",
        messages=[
            ParsedMessage(
                provider_message_id="u1",
                role=Role.USER,
                text="plain text",
                position=0,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="edit",
                        tool_id="tool-1",
                        tool_input={"new_string": "broken \ud83d\udbe0 heading"},
                    )
                ],
            ),
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    session_row = conn.execute("SELECT title FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    block_row = conn.execute("SELECT tool_input FROM blocks WHERE session_id = ?", (session_id,)).fetchone()
    assert session_row["title"] == "Surrogate \ufffd title"
    assert '"new_string":"broken �� heading"' in block_row["tool_input"]


def test_archive_tiers_writer_preserves_chatgpt_branch_variants(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="chatgpt-branch-1",
        title="Regeneration",
        messages=[
            ParsedMessage(
                provider_message_id="u1",
                role=Role.USER,
                text="question",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="question")],
            ),
            ParsedMessage(
                provider_message_id="a-old",
                role=Role.ASSISTANT,
                text="old answer",
                parent_message_provider_id="u1",
                position=1,
                variant_index=0,
                is_active_path=False,
                is_active_leaf=False,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="old answer")],
            ),
            ParsedMessage(
                provider_message_id="a-new",
                role=Role.ASSISTANT,
                text="new answer",
                parent_message_provider_id="u1",
                position=1,
                variant_index=1,
                is_active_path=True,
                is_active_leaf=True,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="new answer")],
            ),
        ],
        active_leaf_message_provider_id="a-new",
    )

    session_id = write_parsed_session_to_archive(conn, session)
    rows = conn.execute(
        """
        SELECT message_id, parent_message_id, position, variant_index, is_active_path, is_active_leaf
        FROM messages
        WHERE session_id = ?
        ORDER BY position, variant_index
        """,
        (session_id,),
    ).fetchall()

    assert [row["message_id"] for row in rows] == [
        "chatgpt-export:chatgpt-branch-1:u1",
        "chatgpt-export:chatgpt-branch-1:a-old",
        "chatgpt-export:chatgpt-branch-1:a-new",
    ]
    assert rows[1]["parent_message_id"] == "chatgpt-export:chatgpt-branch-1:u1"
    assert rows[2]["parent_message_id"] == "chatgpt-export:chatgpt-branch-1:u1"
    assert [(row["is_active_path"], row["is_active_leaf"]) for row in rows] == [(1, 0), (0, 0), (1, 1)]
    assert (
        read_archive_session_envelope(conn, session_id).active_leaf_message_id
        == "chatgpt-export:chatgpt-branch-1:a-new"
    )


def test_archive_tiers_writer_accepts_code_blocks(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="claude-code-1",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                text="print('ok')",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.CODE, text="print('ok')")],
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    block_type = conn.execute("SELECT block_type FROM blocks WHERE session_id = ?", (session_id,)).fetchone()[0]
    assert block_type == "code"


def test_archive_tiers_writer_uses_identity_law_for_messages_without_native_ids(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-generated-ids",
        messages=[
            ParsedMessage(
                provider_message_id="",
                role=Role.USER,
                text="first",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="first")],
            ),
            ParsedMessage(
                provider_message_id="",
                role=Role.ASSISTANT,
                text="second",
                is_active_leaf=True,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="second")],
            ),
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)
    envelope = read_archive_session_envelope(conn, session_id)

    assert [message.message_id for message in envelope.messages] == [
        "codex-session:codex-generated-ids:0.0",
        "codex-session:codex-generated-ids:1.0",
    ]
    assert [block.block_id for message in envelope.messages for block in message.blocks] == [
        "codex-session:codex-generated-ids:0.0:0",
        "codex-session:codex-generated-ids:1.0:0",
    ]


def test_archive_tiers_writer_preserves_session_profile_defaults_with_cost_upsert(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    conn = _connect(db_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-cost-1",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="what's the cost?",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="what's the cost?")],
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)
    upsert_session_profile_costs(
        conn,
        session_id,
        cost_credits=12.34,
        cost_usd=0.056,
        cost_is_estimated=True,
        cost_provenance="estimated",
        priced_with="gpt-5-mini",
        priced_at_ms=1_700_000_006_000,
    )
    conn.close()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    profile = conn.execute(
        """
        SELECT workflow_shape, workflow_shape_method, workflow_shape_confidence, terminal_state,
            terminal_state_method, terminal_state_confidence, duration_ms, substantive_count,
            attachment_count, work_event_count, phase_count, tool_calls_per_minute,
            cost_credits, cost_usd, cost_is_estimated, cost_provenance, priced_with, priced_at_ms,
            search_text
        FROM session_profiles
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    assert profile is not None

    assert dict(profile) == {
        "workflow_shape": None,
        "workflow_shape_method": None,
        "workflow_shape_confidence": None,
        "terminal_state": None,
        "terminal_state_method": None,
        "terminal_state_confidence": None,
        "duration_ms": None,
        "substantive_count": 0,
        "attachment_count": 0,
        "work_event_count": 0,
        "phase_count": 0,
        "tool_calls_per_minute": None,
        "cost_credits": 12.34,
        "cost_usd": 0.056,
        "cost_is_estimated": 1,
        "cost_provenance": "estimated",
        "priced_with": "gpt-5-mini",
        "priced_at_ms": 1_700_000_006_000,
        "search_text": "",
    }


def test_archive_tiers_insight_materialization_upsert_refreshes_shared_state(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-insight-state",
        updated_at="2026-01-01T00:00:02+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="insight state")],
            )
        ],
    )
    session_id = write_parsed_session_to_archive(conn, session)

    first = upsert_insight_materialization(
        conn,
        insight_type="session_profile",
        session_id=session_id,
        materializer_version=1,
        materialized_at_ms=1_767_225_603_000,
        source_updated_at_ms=1_767_225_602_000,
        source_sort_key_ms=1_767_225_602_000,
        input_high_water_mark_ms=1_767_225_602_000,
        input_row_count=1,
    )
    refreshed = upsert_insight_materialization(
        conn,
        insight_type="session_profile",
        session_id=session_id,
        materializer_version=2,
        materialized_at_ms=1_767_225_604_000,
        source_updated_at_ms=1_767_225_602_000,
        source_sort_key_ms=1_767_225_602_000,
        input_high_water_mark_ms=1_767_225_604_000,
        input_row_count=3,
    )

    assert isinstance(first, ArchiveInsightMaterialization)
    assert read_insight_materialization(conn, "session_profile", session_id) == refreshed
    assert refreshed == ArchiveInsightMaterialization(
        insight_type="session_profile",
        session_id=session_id,
        materializer_version=2,
        materialized_at_ms=1_767_225_604_000,
        source_updated_at_ms=1_767_225_602_000,
        source_sort_key_ms=1_767_225_602_000,
        input_high_water_mark_ms=1_767_225_604_000,
        input_row_count=3,
    )


def test_archive_tiers_timeline_insight_rows_have_deterministic_targets(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-timeline-insights",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="implement the plan")],
            ),
            ParsedMessage(
                provider_message_id="m2",
                role=Role.ASSISTANT,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="plan implemented")],
            ),
        ],
    )
    session_id = write_parsed_session_to_archive(conn, session)

    work_event = upsert_session_work_event(
        conn,
        session_id=session_id,
        position=0,
        work_event_type="implementation",
        summary="Implemented archive helper surface",
        confidence=0.875,
        start_index=0,
        end_index=1,
        started_at_ms=1_767_225_600_000,
        ended_at_ms=1_767_225_660_000,
        duration_ms=60_000,
        file_paths=("polylogue/storage/sqlite/archive_tiers/write.py",),
        tools_used=("apply_patch", "pytest"),
        evidence={"message_ids": ["m1", "m2"]},
        inference={"method": "heuristic"},
        search_text="implemented archive helper surface",
    )
    phase = upsert_session_phase(
        conn,
        session_id=session_id,
        position=0,
        start_index=0,
        end_index=1,
        started_at_ms=1_767_225_600_000,
        ended_at_ms=1_767_225_660_000,
        duration_ms=60_000,
        tool_counts={"apply_patch": 1, "pytest": 1},
        word_count=6,
        evidence={"event_positions": [0]},
        inference={"method": "window"},
        search_text="build phase",
    )

    assert work_event == ArchiveSessionWorkEvent(
        event_id=f"{session_id}:work_event:0",
        session_id=session_id,
        position=0,
        work_event_type="implementation",
        summary="Implemented archive helper surface",
        confidence=0.875,
        start_index=0,
        end_index=1,
        started_at_ms=1_767_225_600_000,
        ended_at_ms=1_767_225_660_000,
        duration_ms=60_000,
        file_paths=("polylogue/storage/sqlite/archive_tiers/write.py",),
        tools_used=("apply_patch", "pytest"),
        evidence={"message_ids": ["m1", "m2"]},
        inference={"method": "heuristic"},
        search_text="implemented archive helper surface",
    )
    assert phase == ArchiveSessionPhase(
        phase_id=f"{session_id}:phase:0",
        session_id=session_id,
        position=0,
        start_index=0,
        end_index=1,
        started_at_ms=1_767_225_600_000,
        ended_at_ms=1_767_225_660_000,
        duration_ms=60_000,
        tool_counts={"apply_patch": 1, "pytest": 1},
        word_count=6,
        evidence={"event_positions": [0]},
        inference={"method": "window"},
        search_text="build phase",
    )
    assert read_session_work_events(conn, session_id=session_id) == {0: work_event}
    assert read_session_phases(conn, session_id=session_id) == {0: phase}

    profile = conn.execute(
        "SELECT work_event_count, phase_count FROM session_profiles WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    assert dict(profile) == {"work_event_count": 1, "phase_count": 1}


def test_archive_tiers_session_tags_upsert_normalizes_and_refreshes_scores(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-tags",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="tag me")],
            )
        ],
    )
    session_id = write_parsed_session_to_archive(conn, session)

    first = upsert_session_tag(
        conn,
        session_id=session_id,
        tag=" Archive ",
        tag_source="auto",
        method="heuristic",
        confidence=0.25,
        evidence={"matched": "schema"},
    )
    refreshed = upsert_session_tag(
        conn,
        session_id=session_id,
        tag="archive",
        tag_source="auto",
        method="classifier",
        confidence=0.75,
        evidence={"matched": "v1"},
    )
    assert first.tag == "archive"
    assert refreshed == ArchiveSessionTag(
        session_id=session_id,
        tag="archive",
        tag_source="auto",
        method="classifier",
        confidence=0.75,
        evidence={"matched": "v1"},
    )
    assert read_session_tags(conn, session_id=session_id, tag_source="auto") == {"archive": refreshed}


def test_archive_tiers_writer_materializes_paste_span_from_parser_evidence(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="claude-paste-1",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="pasted body",
                paste_spans=[
                    ParsedPasteEvidence(
                        position=0,
                        start_offset=0,
                        end_offset=11,
                        boundary_state="whole_message_fallback",
                    )
                ],
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="pasted body")],
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    session = conn.execute("SELECT paste_count FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    message = conn.execute(
        "SELECT message_id, has_paste, paste_boundary FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    span = conn.execute(
        """
        SELECT paste_id, position, start_offset, end_offset, boundary_state
        FROM paste_spans
        WHERE message_id = ?
        """,
        (message["message_id"],),
    ).fetchone()
    assert session["paste_count"] == 1
    assert message["has_paste"] == 1
    assert message["paste_boundary"] == "whole_message_fallback"
    envelope = read_archive_session_envelope(conn, session_id)
    assert envelope.messages[0].paste_boundary_state == "whole_message_fallback"
    assert dict(span) == {
        "paste_id": f"{message['message_id']}:0",
        "position": 0,
        "start_offset": 0,
        "end_offset": 11,
        "boundary_state": "whole_message_fallback",
    }


def test_archive_tiers_writer_materializes_supported_session_events(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-events-1",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="summary anchor")],
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="compaction",
                source_message_provider_id="m1",
                timestamp="2026-01-01T00:00:01+00:00",
                payload={"summary": "compressed context"},
            ),
            ParsedSessionEvent(
                event_type="capture_gap",
                timestamp="2026-01-01T00:00:02+00:00",
                payload={"summary": "DOM fallback skipped; richer capture already exists"},
            ),
            ParsedSessionEvent(event_type="turn_context", payload={"cwd": "/tmp"}),
            ParsedSessionEvent(
                event_type="agent_policy",
                timestamp="2026-01-01T00:00:03+00:00",
                payload={"approval": "on-request"},
            ),
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    rows = conn.execute(
        """
        SELECT event_id, source_message_id, position, event_type, summary, occurred_at_ms
        FROM session_events
        WHERE session_id = ?
        ORDER BY position
        """,
        (session_id,),
    ).fetchall()
    assert [dict(row) for row in rows] == [
        {
            "event_id": f"{session_id}:0",
            "source_message_id": f"{session_id}:m1",
            "position": 0,
            "event_type": "compaction",
            "summary": "compressed context",
            "occurred_at_ms": 1_767_225_601_000,
        },
        {
            "event_id": f"{session_id}:1",
            "source_message_id": None,
            "position": 1,
            "event_type": "capture_gap",
            "summary": "DOM fallback skipped; richer capture already exists",
            "occurred_at_ms": 1_767_225_602_000,
        },
    ]
    policies = conn.execute(
        """
        SELECT policy_id, source_message_id, position, approval_policy, sandbox_policy, network_policy, observed_at_ms
        FROM session_agent_policies
        WHERE session_id = ?
        ORDER BY position
        """,
        (session_id,),
    ).fetchall()
    assert [dict(row) for row in policies] == [
        {
            "policy_id": f"{session_id}:2",
            "source_message_id": None,
            "position": 2,
            "approval_policy": "on-request",
            "sandbox_policy": None,
            "network_policy": None,
            "observed_at_ms": 1_767_225_603_000,
        },
    ]


def test_archive_tiers_writer_materializes_provider_usage_events(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-usage-1",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                model_name="gpt-5-codex",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="answer")],
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="m1",
                timestamp="2026-01-01T00:00:03+00:00",
                payload={
                    "type": "token_count",
                    "last_token_usage": {
                        "input_tokens": 11,
                        "cached_input_tokens": 2,
                        "cache_write_tokens": 1,
                        "output_tokens": 3,
                        "reasoning_output_tokens": 4,
                        "total_tokens": 20,
                    },
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 20,
                        "cache_write_tokens": 10,
                        "output_tokens": 30,
                        "reasoning_output_tokens": 40,
                        "total_tokens": 190,
                    },
                    "model_context_window": 200000,
                },
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    usage = conn.execute(
        """
        SELECT usage_event_id, source_message_id, position, provider_event_type,
               last_input_tokens, last_output_tokens, last_cached_input_tokens,
               last_cache_write_tokens, last_reasoning_output_tokens, last_total_tokens,
               total_input_tokens, total_output_tokens, total_cached_input_tokens,
               total_cache_write_tokens, total_reasoning_output_tokens, total_tokens, model_context_window,
               occurred_at_ms
        FROM session_provider_usage_events
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    assert dict(usage) == {
        "usage_event_id": f"{session_id}:usage:0",
        "source_message_id": f"{session_id}:m1",
        "position": 0,
        "provider_event_type": "token_count",
        "last_input_tokens": 11,
        "last_output_tokens": 3,
        "last_cached_input_tokens": 2,
        "last_cache_write_tokens": 1,
        "last_reasoning_output_tokens": 4,
        "last_total_tokens": 20,
        "total_input_tokens": 100,
        "total_output_tokens": 30,
        "total_cached_input_tokens": 20,
        "total_cache_write_tokens": 10,
        "total_reasoning_output_tokens": 40,
        "total_tokens": 190,
        "model_context_window": 200000,
        "occurred_at_ms": 1_767_225_603_000,
    }


def test_provider_usage_events_repair_single_model_usage_rollup(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-usage-rollup-1",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                model_name="gpt-5-codex",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="answer")],
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                payload={
                    "type": "token_count",
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 20,
                        "output_tokens": 30,
                        "reasoning_output_tokens": 40,
                    },
                },
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    usage = conn.execute(
        """
        SELECT model_name, input_tokens, output_tokens, cache_read_tokens,
               cache_write_tokens, cost_provenance
        FROM session_model_usage
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    # Codex reports input inclusive of cached and output inclusive of reasoning.
    # The rollup stores disjoint billing lanes: fresh input = 100 - 20 cached;
    # output = 30 (reasoning is already inside output, not re-added).
    assert dict(usage) == {
        "model_name": "gpt-5-codex",
        "input_tokens": 80,
        "output_tokens": 30,
        "cache_read_tokens": 20,
        "cache_write_tokens": 0,
        "cost_provenance": "origin_reported",
    }


def test_provider_usage_disjoint_lanes_subtracts_cached_and_does_not_re_add_reasoning() -> None:
    from polylogue.storage.sqlite.archive_tiers.write import _provider_usage_disjoint_lanes

    # Codex reports input INCLUSIVE of cached and output INCLUSIVE of reasoning.
    # input=1000 (900 of it cached), output=120 (80 of it reasoning), cache_write=10.
    fresh_input, output, cache_read, cache_write = _provider_usage_disjoint_lanes(1000, 120, 900, 10)
    assert (fresh_input, output, cache_read, cache_write) == (100, 120, 900, 10)
    # Disjoint reconstruction: fresh_input + cache_read == the provider's input,
    # so the cached portion is billed on exactly one lane, not two.
    assert fresh_input + cache_read == 1000
    # Output is passed through unchanged — reasoning is already inside it.
    assert output == 120
    # Guard: if cached somehow exceeds input, fresh input clamps to 0 (never negative).
    assert _provider_usage_disjoint_lanes(20, 10, 30, 0)[0] == 0


def test_provider_usage_events_roll_up_simple_last_usage_when_no_cumulative_total(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-usage-last-only",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                model_name="gpt-5-codex",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="first")],
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                payload={
                    "type": "token_count",
                    "last_token_usage": {
                        "input_tokens": 10,
                        "cached_input_tokens": 2,
                        "cache_write_tokens": 3,
                        "output_tokens": 5,
                        "reasoning_output_tokens": 7,
                    },
                },
            ),
            ParsedSessionEvent(
                event_type="token_count",
                payload={
                    "type": "token_count",
                    "last_token_usage": {"input_tokens": 20, "output_tokens": 6},
                },
            ),
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    usage = conn.execute(
        """
        SELECT input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, cost_provenance
        FROM session_model_usage
        WHERE session_id = ? AND model_name = 'gpt-5-codex'
        """,
        (session_id,),
    ).fetchone()
    # Summed last-usage across two events: input 10+20=30 less cached 2 = 28
    # fresh; output 5+6=11 (reasoning 7 already inside output, not re-added).
    assert dict(usage) == {
        "input_tokens": 28,
        "output_tokens": 11,
        "cache_read_tokens": 2,
        "cache_write_tokens": 3,
        "cost_provenance": "origin_reported",
    }


def test_provider_usage_events_roll_up_session_global_cumulative_to_latest_model(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-usage-multi-model",
        models_used=["gpt-5-codex", "o4-mini"],
        messages=[],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                payload={
                    "type": "token_count",
                    "model": "gpt-5-codex",
                    "total_token_usage": {"input_tokens": 100, "output_tokens": 20},
                },
            ),
            ParsedSessionEvent(
                event_type="token_count",
                payload={
                    "type": "token_count",
                    "model": "o4-mini",
                    "total_token_usage": {"input_tokens": 50, "output_tokens": 10},
                },
            ),
            ParsedSessionEvent(
                event_type="token_count",
                payload={
                    "type": "token_count",
                    "total_token_usage": {"input_tokens": 999, "output_tokens": 999},
                },
            ),
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    rows = conn.execute(
        """
        SELECT model_name, input_tokens, output_tokens
        FROM session_model_usage
        WHERE session_id = ?
        ORDER BY model_name
        """,
        (session_id,),
    ).fetchall()
    # The Codex cumulative total_* is session-global, not per-model. The highest
    # position token_count row with a resolved model is the o4-mini event (the
    # final 999/999 row has no model and 2 models exist, so it is not guessed).
    # Its cumulative already subsumes the earlier gpt-5-codex total, so the
    # rollup attributes the whole session to o4-mini and leaves gpt-5-codex's
    # skeleton row at zero rather than summing 100+50 across models (#2472).
    assert [dict(row) for row in rows] == [
        {"model_name": "gpt-5-codex", "input_tokens": 0, "output_tokens": 0},
        {"model_name": "o4-mini", "input_tokens": 50, "output_tokens": 10},
    ]


def test_provider_usage_cumulative_model_switch_uses_highest_position_not_per_model_sum(
    tmp_path: Path,
) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-usage-model-switch",
        models_used=["gpt-5.3-codex", "gpt-5.4"],
        messages=[],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                payload={
                    "type": "token_count",
                    "model": "gpt-5.3-codex",
                    "total_token_usage": {"input_tokens": 100, "total_tokens": 100},
                },
            ),
            ParsedSessionEvent(
                event_type="token_count",
                payload={
                    "type": "token_count",
                    "model": "gpt-5.3-codex",
                    "total_token_usage": {"input_tokens": 300, "total_tokens": 300},
                },
            ),
            ParsedSessionEvent(
                event_type="token_count",
                payload={
                    "type": "token_count",
                    "model": "gpt-5.4",
                    "total_token_usage": {"input_tokens": 500, "total_tokens": 500},
                },
            ),
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    rows = conn.execute(
        """
        SELECT model_name, input_tokens
        FROM session_model_usage
        WHERE session_id = ?
        ORDER BY model_name
        """,
        (session_id,),
    ).fetchall()
    # The cumulative is session-global. The highest-position event (gpt-5.4,
    # total 500) holds the running total for the WHOLE session, so the rollup is
    # exactly 500 attributed to gpt-5.4 — NOT 300 (gpt-5.3 latest) + 500 = 800
    # split across the two models (#2472).
    assert [dict(row) for row in rows] == [
        {"model_name": "gpt-5.3-codex", "input_tokens": 0},
        {"model_name": "gpt-5.4", "input_tokens": 500},
    ]


def test_provider_usage_per_message_last_usage_still_rolls_up_per_model(tmp_path: Path) -> None:
    # No cumulative total_* anywhere — the Claude-style per-message per-model
    # delta path. These legitimately sum per model and must not regress to a
    # single session-global rollup (#2472).
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-usage-per-message",
        models_used=["model-a", "model-b"],
        messages=[],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                payload={
                    "type": "token_count",
                    "model": "model-a",
                    "last_token_usage": {"input_tokens": 10, "output_tokens": 4},
                },
            ),
            ParsedSessionEvent(
                event_type="token_count",
                payload={
                    "type": "token_count",
                    "model": "model-b",
                    "last_token_usage": {"input_tokens": 7, "output_tokens": 2},
                },
            ),
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    rows = conn.execute(
        """
        SELECT model_name, input_tokens, output_tokens
        FROM session_model_usage
        WHERE session_id = ?
        ORDER BY model_name
        """,
        (session_id,),
    ).fetchall()
    assert [dict(row) for row in rows] == [
        {"model_name": "model-a", "input_tokens": 10, "output_tokens": 4},
        {"model_name": "model-b", "input_tokens": 7, "output_tokens": 2},
    ]


def test_writer_materializes_claude_message_usage_events_without_overriding_priced_rollup(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="claude-usage-ledger",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                model_name="claude-sonnet-4-20250514",
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=2,
                cache_write_tokens=1,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="answer")],
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="message_usage",
                source_message_provider_id="m1",
                payload={
                    "type": "message_usage",
                    "model": "claude-sonnet-4-20250514",
                    "last_token_usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cached_input_tokens": 2,
                        "cache_write_tokens": 1,
                    },
                },
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    event = conn.execute(
        """
        SELECT provider_event_type, model_name, last_input_tokens, last_output_tokens,
               last_cached_input_tokens, last_cache_write_tokens
        FROM session_provider_usage_events
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    assert dict(event) == {
        "provider_event_type": "message_usage",
        "model_name": "claude-sonnet-4-20250514",
        "last_input_tokens": 10,
        "last_output_tokens": 5,
        "last_cached_input_tokens": 2,
        "last_cache_write_tokens": 1,
    }

    rollup = conn.execute(
        """
        SELECT input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, cost_provenance
        FROM session_model_usage
        WHERE session_id = ? AND model_name = 'claude-sonnet-4-20250514'
        """,
        (session_id,),
    ).fetchone()
    assert dict(rollup) == {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_tokens": 2,
        "cache_write_tokens": 1,
        "cost_provenance": "priced",
    }


def test_provider_usage_events_append_preserves_prior_history(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-usage-append",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                model_name="gpt-5-codex",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="first answer")],
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="m1",
                payload={
                    "type": "token_count",
                    "total_token_usage": {"input_tokens": 10, "output_tokens": 5},
                },
            )
        ],
    )
    second = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-usage-append",
        messages=[
            ParsedMessage(
                provider_message_id="m2",
                role=Role.ASSISTANT,
                model_name="gpt-5-codex",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="second answer")],
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="m2",
                payload={
                    "type": "token_count",
                    "total_token_usage": {"input_tokens": 30, "output_tokens": 15},
                },
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, first)
    write_parsed_session_to_archive(conn, second, merge_append=True)

    rows = conn.execute(
        """
        SELECT usage_event_id, source_message_id, position, total_input_tokens, total_output_tokens
        FROM session_provider_usage_events
        WHERE session_id = ?
        ORDER BY position
        """,
        (session_id,),
    ).fetchall()
    assert [dict(row) for row in rows] == [
        {
            "usage_event_id": f"{session_id}:usage:0",
            "source_message_id": f"{session_id}:m1",
            "position": 0,
            "total_input_tokens": 10,
            "total_output_tokens": 5,
        },
        {
            "usage_event_id": f"{session_id}:usage:1",
            "source_message_id": f"{session_id}:m2",
            "position": 1,
            "total_input_tokens": 30,
            "total_output_tokens": 15,
        },
    ]


def test_provider_usage_rollup_skips_append_without_usage_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _connect(tmp_path / "index.db")
    rollup_calls: list[str] = []
    append_rollup_calls: list[tuple[str, int]] = []
    original_rollup = archive_tier_write._aggregate_provider_usage_into_model_usage
    original_append_rollup = archive_tier_write._aggregate_appended_provider_usage_into_model_usage

    def _record_rollup(conn_arg: sqlite3.Connection, session_id_arg: str) -> None:
        rollup_calls.append(session_id_arg)
        original_rollup(conn_arg, session_id_arg)

    def _record_append_rollup(conn_arg: sqlite3.Connection, session_id_arg: str, *, start_position: int) -> None:
        append_rollup_calls.append((session_id_arg, start_position))
        original_append_rollup(conn_arg, session_id_arg, start_position=start_position)

    monkeypatch.setattr(archive_tier_write, "_aggregate_provider_usage_into_model_usage", _record_rollup)
    monkeypatch.setattr(
        archive_tier_write,
        "_aggregate_appended_provider_usage_into_model_usage",
        _record_append_rollup,
    )
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-usage-append-skip",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                model_name="gpt-5-codex",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="first answer")],
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="m1",
                payload={
                    "type": "token_count",
                    "total_token_usage": {"input_tokens": 10, "output_tokens": 5},
                },
            )
        ],
    )
    no_usage_append = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-usage-append-skip",
        messages=[
            ParsedMessage(
                provider_message_id="m2",
                role=Role.ASSISTANT,
                model_name="gpt-5-codex",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="plain append")],
            )
        ],
    )
    usage_append = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-usage-append-skip",
        messages=[
            ParsedMessage(
                provider_message_id="m3",
                role=Role.ASSISTANT,
                model_name="gpt-5-codex",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="usage append")],
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="m3",
                payload={
                    "type": "token_count",
                    "total_token_usage": {"input_tokens": 30, "output_tokens": 15},
                },
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, first)
    assert rollup_calls == [session_id]
    assert append_rollup_calls == []

    write_parsed_session_to_archive(conn, no_usage_append, merge_append=True)
    assert rollup_calls == [session_id]
    assert append_rollup_calls == []

    write_parsed_session_to_archive(conn, usage_append, merge_append=True)
    assert rollup_calls == [session_id]
    assert append_rollup_calls == [(session_id, 1)]

    row = conn.execute(
        """
        SELECT input_tokens, output_tokens, cost_provenance
        FROM session_model_usage
        WHERE session_id = ? AND model_name = 'gpt-5-codex'
        """,
        (session_id,),
    ).fetchone()
    assert dict(row) == {
        "input_tokens": 30,
        "output_tokens": 15,
        "cost_provenance": "origin_reported",
    }


def test_provider_usage_append_incremental_rolls_up_last_usage(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-usage-append-last-only",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                model_name="gpt-5-codex",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="first")],
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="m1",
                payload={
                    "type": "token_count",
                    "last_token_usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "reasoning_output_tokens": 2,
                    },
                },
            )
        ],
    )
    second = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-usage-append-last-only",
        messages=[
            ParsedMessage(
                provider_message_id="m2",
                role=Role.ASSISTANT,
                model_name="gpt-5-codex",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="second")],
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="m2",
                payload={
                    "type": "token_count",
                    "last_token_usage": {
                        "input_tokens": 3,
                        "output_tokens": 4,
                        "reasoning_output_tokens": 1,
                    },
                },
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, first)
    write_parsed_session_to_archive(conn, second, merge_append=True)

    row = conn.execute(
        """
        SELECT input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, cost_provenance
        FROM session_model_usage
        WHERE session_id = ? AND model_name = 'gpt-5-codex'
        """,
        (session_id,),
    ).fetchone()
    # input 10+3=13 (no cached); output 5+4=9 (reasoning 2+1 already inside
    # output, not re-added).
    assert dict(row) == {
        "input_tokens": 13,
        "output_tokens": 9,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cost_provenance": "origin_reported",
    }


def test_provider_usage_append_last_usage_does_not_override_cumulative(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-usage-append-cumulative-first",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                model_name="gpt-5-codex",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="first")],
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="m1",
                payload={
                    "type": "token_count",
                    "model": "gpt-5-codex",
                    "total_token_usage": {"input_tokens": 100, "output_tokens": 40},
                },
            )
        ],
    )
    second = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-usage-append-cumulative-first",
        messages=[
            ParsedMessage(
                provider_message_id="m2",
                role=Role.ASSISTANT,
                model_name="gpt-5-codex",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="second")],
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="m2",
                payload={
                    "type": "token_count",
                    "model": "gpt-5-codex",
                    "last_token_usage": {"input_tokens": 3, "output_tokens": 4},
                },
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, first)
    write_parsed_session_to_archive(conn, second, merge_append=True)

    row = conn.execute(
        """
        SELECT input_tokens, output_tokens, cost_provenance
        FROM session_model_usage
        WHERE session_id = ? AND model_name = 'gpt-5-codex'
        """,
        (session_id,),
    ).fetchone()
    assert dict(row) == {
        "input_tokens": 100,
        "output_tokens": 40,
        "cost_provenance": "origin_reported",
    }


def test_provider_usage_append_model_switch_clears_stale_cumulative(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-usage-append-switch",
        models_used=["gpt-5.3-codex", "gpt-5.4"],
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                model_name="gpt-5.3-codex",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="first")],
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="m1",
                payload={
                    "type": "token_count",
                    "model": "gpt-5.3-codex",
                    "total_token_usage": {"input_tokens": 300, "total_tokens": 300},
                },
            )
        ],
    )
    second = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-usage-append-switch",
        models_used=["gpt-5.3-codex", "gpt-5.4"],
        messages=[
            ParsedMessage(
                provider_message_id="m2",
                role=Role.ASSISTANT,
                model_name="gpt-5.4",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="second")],
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="m2",
                payload={
                    "type": "token_count",
                    "model": "gpt-5.4",
                    "total_token_usage": {"input_tokens": 500, "total_tokens": 500},
                },
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, first)
    write_parsed_session_to_archive(conn, second, merge_append=True)

    rows = conn.execute(
        """
        SELECT model_name, input_tokens
        FROM session_model_usage
        WHERE session_id = ?
        ORDER BY model_name
        """,
        (session_id,),
    ).fetchall()
    # The appended window's latest cumulative (gpt-5.4, 500) is the session-
    # global running total. The earlier gpt-5.3-codex cumulative rollup (300) is
    # now subsumed and must be cleared, not left to be summed back to 800 (#2472).
    assert [dict(row) for row in rows] == [
        {"model_name": "gpt-5.3-codex", "input_tokens": 0},
        {"model_name": "gpt-5.4", "input_tokens": 500},
    ]


def test_reported_costs_skip_message_token_aggregate_on_plain_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _connect(tmp_path / "index.db")
    aggregate_calls: list[str] = []
    original_aggregate = archive_tier_write._aggregate_message_tokens_into_model_usage

    def _record_aggregate(conn_arg: sqlite3.Connection, session_id_arg: str) -> None:
        aggregate_calls.append(session_id_arg)
        original_aggregate(conn_arg, session_id_arg)

    monkeypatch.setattr(archive_tier_write, "_aggregate_message_tokens_into_model_usage", _record_aggregate)
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-cost-append-skip",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                model_name="gpt-5-codex",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="first")],
            )
        ],
    )
    plain_append = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-cost-append-skip",
        messages=[
            ParsedMessage(
                provider_message_id="m2",
                role=Role.ASSISTANT,
                model_name="gpt-5-codex",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="plain append")],
            )
        ],
    )
    token_append = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-cost-append-skip",
        messages=[
            ParsedMessage(
                provider_message_id="m3",
                role=Role.ASSISTANT,
                model_name="gpt-5-codex",
                input_tokens=2,
                output_tokens=3,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="token append")],
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, first)
    assert aggregate_calls == [session_id]

    write_parsed_session_to_archive(conn, plain_append, merge_append=True)
    assert aggregate_calls == [session_id]

    write_parsed_session_to_archive(conn, token_append, merge_append=True)
    assert aggregate_calls == [session_id, session_id]


def test_merge_append_clears_only_existing_active_leaf(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    clear_plan = conn.execute(
        """
        EXPLAIN QUERY PLAN
        UPDATE messages
        SET is_active_leaf = 0
        WHERE session_id = ?
          AND is_active_path = 1
          AND is_active_leaf = 1
        """,
        ("plan-check",),
    ).fetchall()
    assert any("idx_messages_active_leaf" in row["detail"] for row in clear_plan)
    message_count = 200
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-large-append",
        messages=[
            ParsedMessage(
                provider_message_id=f"m{i}",
                role=Role.USER if i % 2 == 0 else Role.ASSISTANT,
                text=f"large append baseline {i}",
                is_active_leaf=i == message_count - 1,
            )
            for i in range(message_count)
        ],
    )
    second = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-large-append",
        messages=[
            ParsedMessage(
                provider_message_id="m200",
                role=Role.ASSISTANT,
                text="large append tail",
                is_active_leaf=True,
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, first)
    before_changes = conn.total_changes
    write_parsed_session_to_archive(conn, second, merge_append=True)
    append_changes = conn.total_changes - before_changes

    rows = conn.execute(
        """
        SELECT native_id, position, is_active_leaf
        FROM messages
        WHERE session_id = ? AND is_active_leaf = 1
        ORDER BY position
        """,
        (session_id,),
    ).fetchall()
    assert [dict(row) for row in rows] == [{"native_id": "m200", "position": 200, "is_active_leaf": 1}]
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0] == 201
    assert append_changes < 80


def test_merge_append_increments_session_counts_without_full_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _connect(tmp_path / "index.db")
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-append-counts",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="typed prompt",
                material_origin=MaterialOrigin.HUMAN_AUTHORED,
            ),
            ParsedMessage(
                provider_message_id="m2",
                role=Role.ASSISTANT,
                text="first answer",
            ),
        ],
    )
    second = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-append-counts",
        messages=[
            ParsedMessage(
                provider_message_id="m3",
                role=Role.USER,
                text="generated context pack",
                material_origin=MaterialOrigin.GENERATED_CONTEXT_PACK,
            ),
            ParsedMessage(
                provider_message_id="m4",
                role=Role.ASSISTANT,
                text="second answer",
                blocks=[ParsedContentBlock(type=BlockType.TOOL_USE, text="call")],
            ),
            ParsedMessage(
                provider_message_id="m5",
                role=Role.TOOL,
                text="tool output",
            ),
        ],
    )

    session_id = write_parsed_session_to_archive(conn, first)

    def _fail_full_refresh(_conn: sqlite3.Connection, _session_id: str) -> None:
        raise AssertionError("merge_append should increment counts instead of rescanning messages")

    monkeypatch.setattr(archive_tier_write, "_refresh_session_counts", _fail_full_refresh)
    write_parsed_session_to_archive(conn, second, merge_append=True)

    row = conn.execute(
        """
        SELECT message_count, word_count, tool_use_count, user_message_count,
               authored_user_message_count, assistant_message_count, tool_message_count,
               user_word_count, authored_user_word_count, assistant_word_count
        FROM sessions
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    assert dict(row) == {
        "message_count": 5,
        "word_count": 11,
        "tool_use_count": 1,
        "user_message_count": 2,
        "authored_user_message_count": 1,
        "assistant_message_count": 2,
        "tool_message_count": 1,
        "user_word_count": 5,
        "authored_user_word_count": 2,
        "assistant_word_count": 4,
    }


def test_merge_append_without_attachments_does_not_refresh_all_attachment_counts(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    unrelated_attachment_count = 120
    for i in range(unrelated_attachment_count):
        write_parsed_session_to_archive(
            conn,
            ParsedSession(
                source_name=Provider.CHATGPT,
                provider_session_id=f"attachment-neighbor-{i}",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=f"neighbor attachment {i}")],
                    )
                ],
                attachments=[
                    ParsedAttachment(
                        provider_attachment_id=f"att-{i}",
                        message_provider_id="m1",
                        name=f"neighbor-{i}.txt",
                        mime_type="text/plain",
                        path=f"neighbor-{i}.txt",
                    )
                ],
            ),
        )
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-append-no-attachments",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="append baseline without attachments",
                is_active_leaf=True,
            )
        ],
    )
    second = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-append-no-attachments",
        messages=[
            ParsedMessage(
                provider_message_id="m2",
                role=Role.ASSISTANT,
                text="append tail without attachments",
                is_active_leaf=True,
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, first)
    before_changes = conn.total_changes
    write_parsed_session_to_archive(conn, second, merge_append=True)
    append_changes = conn.total_changes - before_changes

    assert conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0] == unrelated_attachment_count
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0] == 2
    assert append_changes < 80


def test_full_replace_without_attachments_does_not_refresh_all_attachment_counts(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    unrelated_attachment_count = 120
    for i in range(unrelated_attachment_count):
        write_parsed_session_to_archive(
            conn,
            ParsedSession(
                source_name=Provider.CHATGPT,
                provider_session_id=f"full-replace-attachment-neighbor-{i}",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=f"neighbor attachment {i}")],
                    )
                ],
                attachments=[
                    ParsedAttachment(
                        provider_attachment_id=f"att-{i}",
                        message_provider_id="m1",
                        name=f"neighbor-{i}.txt",
                        mime_type="text/plain",
                        path=f"neighbor-{i}.txt",
                    )
                ],
            ),
        )
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-full-replace-no-attachments",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="baseline without attachments",
            )
        ],
    )
    second = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-full-replace-no-attachments",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="replacement without attachments",
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, first)
    before_changes = conn.total_changes
    write_parsed_session_to_archive(conn, second)
    replace_changes = conn.total_changes - before_changes

    assert conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0] == unrelated_attachment_count
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0] == 1
    assert replace_changes < 80


def test_full_replace_refreshes_removed_attachment_ref_counts(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    att1 = ParsedAttachment(
        provider_attachment_id="att-1",
        message_provider_id="m1",
        name="first.txt",
        mime_type="text/plain",
        path="first.txt",
    )
    att2 = ParsedAttachment(
        provider_attachment_id="att-2",
        message_provider_id="m2",
        name="second.txt",
        mime_type="text/plain",
        path="second.txt",
    )
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="chatgpt-replace-attachment-count",
        messages=[
            ParsedMessage(provider_message_id="m1", role=Role.USER, text="one"),
            ParsedMessage(provider_message_id="m2", role=Role.USER, text="two"),
        ],
        attachments=[att1, att2],
    )
    replacement = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="chatgpt-replace-attachment-count",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="one updated")],
        attachments=[att1],
    )

    session_id = write_parsed_session_to_archive(conn, session)
    att2_id = archive_tier_write._attachment_id(session_id, att2)
    write_parsed_session_to_archive(conn, replacement)

    assert conn.execute("SELECT ref_count FROM attachments WHERE attachment_id = ?", (att2_id,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM attachment_refs WHERE attachment_id = ?", (att2_id,)).fetchone()[0] == 0


def test_provider_usage_rollup_clears_stale_message_pricing(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-usage-priced-repair",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                model_name="gpt-4o",
                input_tokens=10,
                output_tokens=5,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="priced answer")],
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                payload={
                    "type": "token_count",
                    "model": "gpt-4o",
                    "total_token_usage": {"input_tokens": 1_000, "output_tokens": 500},
                },
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    row = conn.execute(
        """
        SELECT model_name, input_tokens, output_tokens, cost_provenance,
               cost_usd, priced_with, priced_at_ms
        FROM session_model_usage
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    assert dict(row) == {
        "model_name": "gpt-4o",
        "input_tokens": 1_000,
        "output_tokens": 500,
        "cost_provenance": "origin_reported",
        "cost_usd": None,
        "priced_with": None,
        "priced_at_ms": None,
    }


def test_archive_tiers_writer_records_unresolved_parent_session_link(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="child-session",
        parent_session_provider_id="parent-session",
        branch_type=BranchType.SIDECHAIN,
        updated_at="2026-01-01T00:00:03+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="child")],
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    row = conn.execute(
        """
        SELECT src_session_id, dst_origin, dst_native_id, resolved_dst_session_id, link_type, status, method, confidence,
            evidence_json, observed_at_ms
        FROM session_links
        WHERE src_session_id = ?
        """,
        (session_id,),
    ).fetchone()
    assert dict(row) == {
        "src_session_id": "claude-code-session:child-session",
        "dst_origin": "claude-code-session",
        "dst_native_id": "parent-session",
        "resolved_dst_session_id": None,
        "link_type": "sidechain",
        "status": None,
        "method": "parser-parent",
        "confidence": 1.0,
        "evidence_json": '{"parent_session_provider_id":"parent-session"}',
        "observed_at_ms": 1_767_225_603_000,
    }


def test_archive_tiers_writer_resolves_parent_link_when_parent_already_exists(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    parent = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="parent-session",
        updated_at="2026-01-01T00:00:01+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="p1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="parent")],
            )
        ],
    )
    child = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="child-session",
        parent_session_provider_id="parent-session",
        branch_type=BranchType.SIDECHAIN,
        updated_at="2026-01-01T00:00:02+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="c1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="child")],
            )
        ],
    )

    parent_id = write_parsed_session_to_archive(conn, parent)
    child_id = write_parsed_session_to_archive(conn, child)

    child_row = conn.execute(
        "SELECT parent_session_id, root_session_id, branch_type FROM sessions WHERE session_id = ?",
        (child_id,),
    ).fetchone()
    link_row = conn.execute(
        "SELECT resolved_dst_session_id, status FROM session_links WHERE src_session_id = ?",
        (child_id,),
    ).fetchone()
    thread_rows = conn.execute(
        """
        SELECT thread_id, session_id, position
        FROM thread_sessions
        WHERE thread_id = ?
        ORDER BY position
        """,
        (parent_id,),
    ).fetchall()

    assert dict(child_row) == {
        "parent_session_id": parent_id,
        "root_session_id": parent_id,
        "branch_type": "sidechain",
    }
    assert dict(link_row) == {"resolved_dst_session_id": parent_id, "status": None}
    assert [dict(row) for row in thread_rows] == [
        {"thread_id": parent_id, "session_id": parent_id, "position": 0},
        {"thread_id": parent_id, "session_id": child_id, "position": 1},
    ]
    assert conn.execute("SELECT session_count FROM threads WHERE thread_id = ?", (parent_id,)).fetchone()[0] == 2
    plan_rows = conn.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT session_id
        FROM sessions
        WHERE root_session_id = ? OR session_id = ?
        ORDER BY sort_key_ms IS NULL, sort_key_ms, session_id
        """,
        (parent_id, parent_id),
    ).fetchall()
    plan = "\n".join(str(row[3]) for row in plan_rows)
    assert "idx_sessions_root" in plan
    assert "SCAN sessions" not in plan


def test_refresh_thread_fast_path_keeps_current_thread_membership(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    parent = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="parent-fast-path",
        updated_at="2026-01-01T00:00:01+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="p1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="parent")],
            )
        ],
    )
    child = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="child-fast-path",
        parent_session_provider_id="parent-fast-path",
        branch_type=BranchType.SIDECHAIN,
        updated_at="2026-01-01T00:00:02+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="c1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="child")],
            )
        ],
    )

    parent_id = write_parsed_session_to_archive(conn, parent)
    child_id = write_parsed_session_to_archive(conn, child)
    before = conn.execute(
        """
        SELECT session_id, position
        FROM thread_sessions
        WHERE thread_id = ?
        ORDER BY position
        """,
        (parent_id,),
    ).fetchall()
    statements: list[str] = []
    conn.set_trace_callback(statements.append)

    archive_tier_write._refresh_thread(conn, parent_id)

    conn.set_trace_callback(None)
    after = conn.execute(
        """
        SELECT session_id, position
        FROM thread_sessions
        WHERE thread_id = ?
        ORDER BY position
        """,
        (parent_id,),
    ).fetchall()
    mutating_thread_statements = [
        stmt for stmt in statements if ("DELETE FROM thread_sessions" in stmt or "INSERT INTO thread_sessions" in stmt)
    ]
    assert [(row[0], row[1]) for row in before] == [(parent_id, 0), (child_id, 1)]
    assert [(row[0], row[1]) for row in after] == [(parent_id, 0), (child_id, 1)]
    assert mutating_thread_statements == []


def test_refresh_thread_appends_suffix_without_rebuilding_membership(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    parent = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="parent-suffix",
        updated_at="2026-01-01T00:00:01+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="p1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="parent")],
            )
        ],
    )
    first_child = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="child-suffix-1",
        parent_session_provider_id="parent-suffix",
        branch_type=BranchType.SIDECHAIN,
        updated_at="2026-01-01T00:00:02+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="c1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="child one")],
            )
        ],
    )
    second_child = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="child-suffix-2",
        parent_session_provider_id="parent-suffix",
        branch_type=BranchType.SIDECHAIN,
        updated_at="2026-01-01T00:00:03+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="c2",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="child two")],
            )
        ],
    )

    parent_id = write_parsed_session_to_archive(conn, parent)
    first_child_id = write_parsed_session_to_archive(conn, first_child)
    statements: list[str] = []
    conn.set_trace_callback(statements.append)

    second_child_id = write_parsed_session_to_archive(conn, second_child)

    conn.set_trace_callback(None)
    thread_rows = conn.execute(
        """
        SELECT session_id, position
        FROM thread_sessions
        WHERE thread_id = ?
        ORDER BY position
        """,
        (parent_id,),
    ).fetchall()
    parent_thread_deletes = [
        stmt for stmt in statements if "DELETE FROM thread_sessions" in stmt and f"'{parent_id}'" in stmt
    ]
    assert [(row[0], row[1]) for row in thread_rows] == [
        (parent_id, 0),
        (first_child_id, 1),
        (second_child_id, 2),
    ]
    assert parent_thread_deletes == []


def test_archive_tiers_writer_resolves_existing_child_link_when_parent_arrives_later(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    child = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="child-first",
        parent_session_provider_id="parent-later",
        branch_type=BranchType.CONTINUATION,
        updated_at="2026-01-01T00:00:02+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="c1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="child")],
            )
        ],
    )
    parent = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="parent-later",
        updated_at="2026-01-01T00:00:01+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="p1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="parent")],
            )
        ],
    )

    child_id = write_parsed_session_to_archive(conn, child)
    assert conn.execute("SELECT thread_id FROM threads WHERE thread_id = ?", (child_id,)).fetchone()[0] == child_id

    parent_id = write_parsed_session_to_archive(conn, parent)

    child_row = conn.execute(
        "SELECT parent_session_id, root_session_id, branch_type FROM sessions WHERE session_id = ?",
        (child_id,),
    ).fetchone()
    link_row = conn.execute(
        "SELECT resolved_dst_session_id, status FROM session_links WHERE src_session_id = ?",
        (child_id,),
    ).fetchone()
    stale_child_thread = conn.execute("SELECT thread_id FROM threads WHERE thread_id = ?", (child_id,)).fetchone()
    thread_rows = conn.execute(
        """
        SELECT thread_id, session_id, position
        FROM thread_sessions
        WHERE thread_id = ?
        ORDER BY position
        """,
        (parent_id,),
    ).fetchall()

    assert dict(child_row) == {
        "parent_session_id": parent_id,
        "root_session_id": parent_id,
        "branch_type": "continuation",
    }
    assert dict(link_row) == {"resolved_dst_session_id": parent_id, "status": None}
    assert stale_child_thread is None
    assert [dict(row) for row in thread_rows] == [
        {"thread_id": parent_id, "session_id": parent_id, "position": 0},
        {"thread_id": parent_id, "session_id": child_id, "position": 1},
    ]


def test_graph_resolve_records_late_parent_substage_timings(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    child = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="child-timing",
        parent_session_provider_id="parent-timing",
        branch_type=BranchType.CONTINUATION,
        updated_at="2026-01-01T00:00:02+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="shared",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="shared prefix")],
            ),
            ParsedMessage(
                provider_message_id="tail",
                role=Role.ASSISTANT,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="child tail")],
            ),
        ],
    )
    parent = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="parent-timing",
        updated_at="2026-01-01T00:00:01+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="shared",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="shared prefix")],
            )
        ],
    )

    write_parsed_session_to_archive(conn, child)
    timings: dict[str, float] = {}

    write_parsed_session_to_archive(
        conn,
        parent,
        stage_timings_s=timings,
        stage_timing_prefix="append",
    )

    expected_substages = {
        "append.index.graph_resolve.root_init",
        "append.index.graph_resolve.outbound_links",
        "append.index.graph_resolve.inbound_lookup",
        "append.index.graph_resolve.root_current_check",
        "append.index.graph_resolve.reextract_prefix_tails",
        "append.index.graph_resolve.reextract_prefix_tails.edge_lookup",
        "append.index.graph_resolve.reextract_prefix_tails.parent_composed",
        "append.index.graph_resolve.reextract_prefix_tails.child_composed",
        "append.index.graph_resolve.reextract_prefix_tails.signature_compare",
        "append.index.graph_resolve.reextract_prefix_tails.provider_usage_tail",
        "append.index.graph_resolve.reextract_prefix_tails.dependent_delete",
        "append.index.graph_resolve.reextract_prefix_tails.message_delete",
        "append.index.graph_resolve.reextract_prefix_tails.edge_update",
        "append.index.graph_resolve.reextract_prefix_tails.count_refresh",
        "append.index.graph_resolve.projection_refresh",
        "append.index.graph_resolve.thread_refresh",
    }
    assert expected_substages <= timings.keys()
    assert "append.index.graph_resolve" in timings


def test_own_db_signatures_uses_session_scoped_block_lookup(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    plan_rows = conn.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT m.message_id, m.position, m.role,
               b.block_type, b.text, b.tool_name, b.tool_input
        FROM messages m
        LEFT JOIN blocks b ON b.session_id = m.session_id AND b.message_id = m.message_id
        WHERE m.session_id = ? AND m.variant_index = 0
        ORDER BY m.position, b.position
        """,
        ("session:planner",),
    ).fetchall()
    plan = "\n".join(str(row[3]) for row in plan_rows)

    assert "idx_blocks_session_position" in plan
    assert "sqlite_autoindex_blocks_2" not in plan


def test_graph_resolve_shares_projection_refresh_seen_set_for_late_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _connect(tmp_path / "index.db")
    grandparent = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="grandparent-seen",
        updated_at="2026-01-01T00:00:00+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="gp",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="grandparent")],
            )
        ],
    )
    parent = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="parent-seen",
        parent_session_provider_id="grandparent-seen",
        branch_type=BranchType.CONTINUATION,
        updated_at="2026-01-01T00:00:01+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="p",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="parent")],
            )
        ],
    )
    children = [
        ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id=f"child-seen-{position}",
            parent_session_provider_id="parent-seen",
            branch_type=BranchType.SIDECHAIN,
            updated_at=f"2026-01-01T00:00:0{position + 2}+00:00",
            messages=[
                ParsedMessage(
                    provider_message_id=f"c{position}",
                    role=Role.ASSISTANT,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text=f"child {position}")],
                )
            ],
        )
        for position in range(3)
    ]

    write_parsed_session_to_archive(conn, grandparent)
    for child in children:
        write_parsed_session_to_archive(conn, child)

    original_refresh = archive_tier_write._refresh_session_projection
    seen_ids: list[int] = []

    def wrapped_refresh(session_conn: sqlite3.Connection, session_id: str, *, seen: set[str]) -> None:
        seen_ids.append(id(seen))
        original_refresh(session_conn, session_id, seen=seen)

    monkeypatch.setattr(archive_tier_write, "_refresh_session_projection", wrapped_refresh)

    write_parsed_session_to_archive(conn, parent)

    assert len(set(seen_ids)) == 1


def test_archive_tiers_writer_materializes_repo_and_commit_edges(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="repo-session",
        git_branch="feature/archive",
        git_repository_url="https://github.com/Sinity/polylogue.git",
        git_commit_hash="0123456789abcdef0123456789abcdef01234567",
        working_directories=["/realm/project/polylogue"],
        updated_at="2026-01-01T00:00:05+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="repo edge")],
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    repo = conn.execute(
        """
        SELECT origin_url, root_path, repo_id, repo_name, first_seen_at_ms, last_seen_at_ms
        FROM repos
        """
    ).fetchone()
    session_repo = conn.execute(
        """
        SELECT session_id, repo_id, branch_name, observed_at_ms
        FROM session_repos
        """
    ).fetchone()
    commit = conn.execute(
        """
        SELECT session_id, commit_sha, repo_id, detection_type, method, confidence,
            evidence_json, created_at_ms
        FROM session_commits
        """
    ).fetchone()

    assert dict(repo) == {
        "origin_url": "https://github.com/Sinity/polylogue.git",
        "root_path": "/realm/project/polylogue",
        "repo_id": "https://github.com/Sinity/polylogue.git\x1f/realm/project/polylogue",
        "repo_name": "polylogue",
        "first_seen_at_ms": 1_767_225_605_000,
        "last_seen_at_ms": 1_767_225_605_000,
    }
    assert dict(session_repo) == {
        "session_id": session_id,
        "repo_id": "https://github.com/Sinity/polylogue.git\x1f/realm/project/polylogue",
        "branch_name": "feature/archive",
        "observed_at_ms": 1_767_225_605_000,
    }
    assert dict(commit) == {
        "session_id": session_id,
        "commit_sha": "0123456789abcdef0123456789abcdef01234567",
        "repo_id": "https://github.com/Sinity/polylogue.git\x1f/realm/project/polylogue",
        "detection_type": "explicit_ref",
        "method": "parser-git-meta",
        "confidence": 1.0,
        "evidence_json": (
            '{"git_branch":"feature/archive",'
            '"git_repository_url":"https://github.com/Sinity/polylogue.git",'
            '"root_path":"/realm/project/polylogue"}'
        ),
        "created_at_ms": 1_767_225_605_000,
    }


def test_archive_tiers_writer_replacement_clears_old_projection_rows(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="replace-session",
        git_branch="feature/archive",
        git_repository_url="https://github.com/Sinity/polylogue.git",
        git_commit_hash="0123456789abcdef0123456789abcdef01234567",
        working_directories=["/realm/project/polylogue"],
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="with old projections")],
            )
        ],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="att-1",
                message_provider_id="m1",
                name="report.pdf",
                mime_type="application/pdf",
                path="report.pdf",
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="compaction",
                source_message_provider_id="m1",
                payload={"summary": "old event"},
                timestamp="2026-01-01T00:00:00+00:00",
            )
        ],
    )
    second = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="replace-session",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="replacement")],
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, first)
    write_parsed_session_to_archive(conn, second)

    envelope = read_archive_session_envelope(conn, session_id)
    stale_counts = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table} WHERE session_id = ?", (session_id,)).fetchone()[0]
        for table in (
            "attachment_refs",
            "session_events",
            "session_working_dirs",
            "session_repos",
            "session_commits",
        )
    }
    session_row = conn.execute(
        """
        SELECT git_branch, git_repository_url, commit_hash
        FROM sessions
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()

    assert envelope.orphan_attachments == ()
    assert envelope.messages[0].attachments == ()
    assert stale_counts == {
        "attachment_refs": 0,
        "session_events": 0,
        "session_working_dirs": 0,
        "session_repos": 0,
        "session_commits": 0,
    }
    assert dict(session_row) == {"git_branch": None, "git_repository_url": None, "commit_hash": None}
    assert search_archive_blocks(conn, "old") == []
    assert search_archive_blocks(conn, "replacement") == [f"{session_id}:m1:0"]
    assert (
        conn.execute(
            """
        SELECT COUNT(*)
        FROM sqlite_master
        WHERE type = 'trigger'
          AND name IN ('messages_fts_ai', 'messages_fts_ad', 'messages_fts_au')
        """
        ).fetchone()[0]
        == 3
    )
    assert (
        conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]
        == conn.execute("SELECT COUNT(*) FROM blocks WHERE search_text != ''").fetchone()[0]
    )
    for table, index_name in (
        ("session_events", "idx_session_events_source_message"),
        ("session_agent_policies", "idx_session_agent_policies_source_message"),
        ("session_provider_usage_events", "idx_session_provider_usage_events_source_message"),
    ):
        assert any(row["name"] == index_name for row in conn.execute(f"PRAGMA index_list({table})"))


def test_archive_tiers_writer_materializes_attachments_and_refs(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="chatgpt-attachment",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="see attachment")],
            )
        ],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="att-1",
                message_provider_id="m1",
                name="report.pdf",
                mime_type="application/pdf",
                size_bytes=1234,
                path="report.pdf",
                provider_file_id="file-1",
                provider_drive_id="drive-1",
                upload_origin="drive",
                caption="report",
                source_url="https://example.test/report.pdf",
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)
    attachment_hash = hashlib.sha256()
    for part in (
        "attachment",
        "att-1",
        "file-1",
        "drive-1",
        "report.pdf",
        "report.pdf",
        "application/pdf",
        "1234",
    ):
        attachment_hash.update(part.encode("utf-8", errors="surrogatepass"))
        attachment_hash.update(b"\0")
    attachment_id = attachment_hash.hexdigest()
    message_id = f"{session_id}:m1"

    attachment = conn.execute(
        """
        SELECT attachment_id, display_name, media_type, byte_count, ref_count
        FROM attachments
        WHERE attachment_id = ?
        """,
        (attachment_id,),
    ).fetchone()
    ref = conn.execute(
        """
        SELECT ref_id, attachment_id, session_id, message_id, position, upload_origin, source_url, caption
        FROM attachment_refs
        WHERE attachment_id = ?
        """,
        (attachment_id,),
    ).fetchone()

    assert dict(attachment) == {
        "attachment_id": attachment_id,
        "display_name": "report.pdf",
        "media_type": "application/pdf",
        "byte_count": 1234,
        "ref_count": 1,
    }
    ref_dict = dict(ref)
    assert {
        key: ref_dict[key]
        for key in ("attachment_id", "session_id", "message_id", "upload_origin", "source_url", "caption")
    } == {
        "attachment_id": attachment_id,
        "session_id": session_id,
        "message_id": message_id,
        "upload_origin": "drive",
        "source_url": "https://example.test/report.pdf",
        "caption": "report",
    }
    native_ids = {
        (row["id_kind"], row["native_id"])
        for row in conn.execute(
            "SELECT id_kind, native_id FROM attachment_native_ids WHERE ref_id = ?",
            (ref_dict["ref_id"],),
        ).fetchall()
    }
    assert native_ids == {
        ("attachment", "att-1"),
        ("file", "file-1"),
        ("drive", "drive-1"),
        ("url", "https://example.test/report.pdf"),
    }


# ---------------------------------------------------------------------------
# ITEM 1: sessions.instructions_text round-trip
# ---------------------------------------------------------------------------


def test_instructions_text_roundtrip(tmp_path: Path) -> None:
    """ParsedSession.instructions_text written by the writer and read back."""
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="instr-session-1",
        title="Instructions test",
        created_at="2026-01-01T00:00:00+00:00",
        instructions_text="Always reply in haiku.",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="hello",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hello")],
            ),
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)
    envelope = read_archive_session_envelope(conn, session_id)

    assert envelope.instructions_text == "Always reply in haiku."


def test_instructions_text_none_when_absent(tmp_path: Path) -> None:
    """instructions_text is None for sessions that carry no instructions."""
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="no-instr-1",
        title="No instructions",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="hi",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hi")],
            ),
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)
    envelope = read_archive_session_envelope(conn, session_id)

    assert envelope.instructions_text is None


# ---------------------------------------------------------------------------
# ITEM 2: sessions.title_source round-trip
# ---------------------------------------------------------------------------


def test_title_source_roundtrip(tmp_path: Path) -> None:
    """TitleSource enum values survive the write → read cycle unchanged."""
    conn = _connect(tmp_path / "index.db")
    for title_source, expected in (
        (TitleSource.ORIGIN, "origin"),
        (TitleSource.HEURISTIC, "heuristic"),
        (TitleSource.UNKNOWN, "unknown"),
        (TitleSource.PATH, "path"),
        (TitleSource.USER, "user"),
    ):
        session = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=f"ts-{expected}",
            title=f"Title-{expected}",
            title_source=title_source,
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.USER,
                    text="x",
                    position=0,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="x")],
                ),
            ],
        )
        session_id = write_parsed_session_to_archive(conn, session)
        envelope = read_archive_session_envelope(conn, session_id)
        assert envelope.title_source == expected, f"expected title_source={expected!r}, got {envelope.title_source!r}"


def test_title_source_none_when_absent(tmp_path: Path) -> None:
    """title_source is NULL when no parser sets it."""
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="ts-none",
        title="No source",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="x",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="x")],
            ),
        ],
    )
    session_id = write_parsed_session_to_archive(conn, session)
    envelope = read_archive_session_envelope(conn, session_id)
    assert envelope.title_source is None


# ---------------------------------------------------------------------------
# ITEM 3: session_agent_policies round-trip
# ---------------------------------------------------------------------------


def test_agent_policy_roundtrip(tmp_path: Path) -> None:
    """agent_policy ParsedSessionEvent written by the writer and read back."""
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="policy-session-1",
        title="Policy test",
        created_at="2026-01-01T00:00:00+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="do it",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="do it")],
            ),
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="agent_policy",
                timestamp="2026-01-01T00:00:01+00:00",
                payload={
                    "approval_policy": "never",
                    "sandbox_policy": "danger-full-access",
                    "network_policy": "true",
                },
            ),
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)
    policies = read_session_agent_policies(conn, session_id)

    assert len(policies) == 1
    policy = policies[0]
    assert isinstance(policy, ArchiveAgentPolicy)
    assert policy.approval_policy == "never"
    assert policy.sandbox_policy == "danger-full-access"
    assert policy.network_policy == "true"
    assert policy.position == 0


def test_agent_policy_absent_when_no_events(tmp_path: Path) -> None:
    """read_session_agent_policies returns [] for sessions without policy events."""
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="no-policy-session",
        title="No policy",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="hi",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hi")],
            ),
        ],
    )
    session_id = write_parsed_session_to_archive(conn, session)
    assert read_session_agent_policies(conn, session_id) == []


# ---------------------------------------------------------------------------
# Regression: ingest_flags written as auto-tags at write time (#1764)
# ---------------------------------------------------------------------------


def test_ingest_flags_written_as_auto_tags(tmp_path: Path) -> None:
    """ParsedSession.ingest_flags are persisted as auto-tags during write.

    The storage layer calls _write_ingest_flag_tags inside the same transaction
    as the session row, so the tags are always present when the session is
    readable.  This is the durable persistence path for parser-level quality
    flags (e.g. ``degraded:brain-metadata-fragment`` for Antigravity brain-
    artifact fallback sessions, issue #1764).
    """
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="flagged-session",
        title="Flagged session",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                text="artifact body",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="artifact body")],
            ),
        ],
        ingest_flags=["degraded:brain-metadata-fragment"],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    tags = read_session_tags(conn, session_id=session_id, tag_source="auto")
    assert "degraded:brain-metadata-fragment" in tags, (
        f"Expected degraded:brain-metadata-fragment auto-tag, got: {list(tags)}"
    )
    tag = tags["degraded:brain-metadata-fragment"]
    assert isinstance(tag, ArchiveSessionTag)
    assert tag.tag_source == "auto"
    assert tag.method == "parser"


def test_session_kind_is_persisted_and_read_back(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="temporary-session",
        title="Temporary session",
        session_kind=SessionKind.TEMPORARY,
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="temporary text",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="temporary text")],
            )
        ],
        ingest_flags=["capture:temporary-chat"],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    row = conn.execute("SELECT session_kind FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    assert row["session_kind"] == "temporary"
    envelope = read_archive_session_envelope(conn, session_id)
    assert envelope.session_kind == SessionKind.TEMPORARY.value


def test_ingest_flags_empty_writes_no_auto_tags(tmp_path: Path) -> None:
    """Sessions with no ingest_flags do not gain spurious auto-tags."""
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="clean-session",
        title="Clean session",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="hello",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hello")],
            ),
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    auto_tags = read_session_tags(conn, session_id=session_id, tag_source="auto")
    assert auto_tags == {}, f"Expected no auto-tags for clean session, got: {list(auto_tags)}"


def test_ingest_flags_re_ingest_is_idempotent(tmp_path: Path) -> None:
    """Re-ingesting a session with ingest_flags does not duplicate auto-tags."""
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="re-ingest-session",
        title="Re-ingest test",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                text="body",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="body")],
            ),
        ],
        ingest_flags=["degraded:brain-metadata-fragment"],
    )

    write_parsed_session_to_archive(conn, session)
    write_parsed_session_to_archive(conn, session)

    auto_tags = read_session_tags(conn, session_id=write_parsed_session_to_archive(conn, session), tag_source="auto")
    assert len(auto_tags) == 1, (
        f"Expected exactly one auto-tag after re-ingest, got {len(auto_tags)}: {list(auto_tags)}"
    )
