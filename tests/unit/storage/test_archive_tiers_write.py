from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import TitleSource
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedPasteEvidence,
    ParsedSession,
    ParsedSessionEvent,
)
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
from polylogue.types import BlockType, Provider


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


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
    action = conn.execute("SELECT tool_command, output_text FROM actions").fetchone()
    assert dict(action) == {"tool_command": "pytest -q", "output_text": "checks passed"}
    assert search_archive_blocks(conn, "focused") == ["codex-session:codex-session-1:u1:0"]


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
        phase_type="build",
        confidence=0.75,
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
        phase_type="build",
        confidence=0.75,
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
    user_tag = upsert_session_tag(
        conn,
        session_id=session_id,
        tag="Pinned",
        tag_source="user",
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
    assert user_tag == ArchiveSessionTag(
        session_id=session_id,
        tag="pinned",
        tag_source="user",
        method=None,
        confidence=None,
        evidence=None,
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
        "SELECT message_id, has_paste FROM messages WHERE session_id = ?",
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
            ParsedSessionEvent(event_type="turn_context", payload={"cwd": "/tmp"}),
            ParsedSessionEvent(
                event_type="agent_policy",
                timestamp="2026-01-01T00:00:02+00:00",
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
            "policy_id": f"{session_id}:1",
            "source_message_id": None,
            "position": 1,
            "approval_policy": "on-request",
            "sandbox_policy": None,
            "network_policy": None,
            "observed_at_ms": 1_767_225_602_000,
        },
    ]


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
