from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.query.unit_results import query_unit_envelope, query_unit_request
from polylogue.demo import seed_demo_archive, verify_demo_archive
from polylogue.scenarios import (
    DEMO_CHATGPT_SESSION_ID,
    DEMO_CLAUDE_AI_TEMPORARY_SESSION_ID,
    DEMO_CLAUDE_CODE_LINEAGE_SIDECHAIN_SESSION_ID,
    DEMO_CLAUDE_CODE_SESSION_ID,
    DEMO_EMBEDDING_PROSE_SESSION_ID,
    DEMO_SESSION_IDS,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


@pytest.mark.asyncio
async def test_seed_demo_archive_creates_ready_queryable_archive(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"

    seed = await seed_demo_archive(archive_root, force=True, with_overlays=True)
    verify = verify_demo_archive(archive_root, require_overlays=True)

    assert seed.archive_root == archive_root
    assert seed.session_count == len(DEMO_SESSION_IDS)
    assert seed.message_count >= 35
    assert seed.session_ids == tuple(sorted(DEMO_SESSION_IDS))
    assert seed.overlays_seeded is True
    assert seed.assertion_count >= 4
    assert seed.construct_coverage
    assert all(row.ok for row in seed.construct_coverage)

    assert verify.ok is True
    assert verify.session_count == len(DEMO_SESSION_IDS)
    assert verify.message_count >= 35
    assert DEMO_CLAUDE_CODE_SESSION_ID in verify.query_hits
    assert verify.overlays_present is True
    assert verify.absolute_path_leaks == ()
    assert verify.construct_coverage
    assert all(row.ok for row in verify.construct_coverage)
    assert verify.problems == ()

    with ArchiveStore.open_existing(archive_root) as archive:
        delegation_rows = query_unit_envelope(
            archive,
            query_unit_request(
                expression="delegations where parent:demo-lineage-parent AND mapping_state:resolved",
                limit=10,
            ),
        )
        [delegation_item] = delegation_rows.items
        delegation = delegation_item.model_dump(mode="json")
        assert delegation["parent_session_id"] == "codex-session:demo-lineage-parent"
        assert delegation["child_session_id"] == "codex-session:demo-lineage-subagent"
        assert delegation["mapping_state"] == "resolved"
        assert delegation["evidence_basis"] == "action"
        assert delegation["instruction_preview"] == "Inspect the demo lineage child and report caveats."
        expected_instruction = "Inspect the demo lineage child and report caveats."
        assert delegation["instruction_sha256"] == hashlib.sha256(expected_instruction.encode()).hexdigest()
        assert delegation["instruction_truncated"] is False
        instruction_block_id = delegation["instruction_tool_use_block_id"]
        assert isinstance(instruction_block_id, str)
        card = archive.get_delegation_card(instruction_tool_use_block_id=instruction_block_id)
        assert card is not None
        assert card.instruction == expected_instruction
        assert card.dispatch_result == "Subagent completed. Session: codex-session:demo-lineage-subagent"
        assert card.child_excerpt == (
            "Subagent report: lineage fixture has a parent, a fork, and a resolved child link."
        )

    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute("ATTACH DATABASE ? AS source", (str(archive_root / "source.db"),))
        links = conn.execute("SELECT link_type, inheritance FROM session_links ORDER BY src_session_id").fetchall()
        temporary_sessions = conn.execute("SELECT session_id FROM sessions WHERE session_kind = 'temporary'").fetchall()
        capture_gap_events = conn.execute(
            "SELECT session_id, summary FROM session_events WHERE event_type = 'capture_gap'"
        ).fetchall()
        chatgpt_raw_rows = conn.execute(
            """
            SELECT COUNT(*)
            FROM source.raw_sessions
            WHERE origin = 'chatgpt-export'
              AND native_id = ?
            """,
            (DEMO_CHATGPT_SESSION_ID.split(":", maxsplit=1)[1],),
        ).fetchone()[0]
        chatgpt_session_rows = conn.execute(
            """
            SELECT COUNT(*)
            FROM sessions AS s
            JOIN source.raw_sessions AS r
              ON r.raw_id = s.raw_id
            WHERE s.session_id = ?
            """,
            (DEMO_CHATGPT_SESSION_ID,),
        ).fetchone()[0]
        compaction_events = conn.execute(
            "SELECT session_id, summary FROM session_events WHERE event_type = 'compaction'"
        ).fetchall()
        sidechain_sessions = conn.execute("SELECT session_id FROM sessions WHERE branch_type = 'sidechain'").fetchall()
        subagent_snapshots = conn.execute(
            "SELECT COUNT(*) FROM session_context_snapshots WHERE boundary = 'subagent_start'"
        ).fetchone()[0]
        subagent_runs = conn.execute("SELECT COUNT(*) FROM session_runs WHERE role = 'subagent'").fetchone()[0]
        unfinished_terminal_states = conn.execute(
            "SELECT terminal_state, COUNT(*) FROM session_profiles GROUP BY terminal_state"
        ).fetchall()
    with sqlite3.connect(archive_root / "embeddings.db") as conn:
        embedding_rows = conn.execute(
            "SELECT COUNT(*) FROM message_embeddings_meta WHERE model = 'demo-synthetic-embedding'"
        ).fetchone()[0]
        embedding_status = conn.execute(
            "SELECT session_id, message_count_embedded, needs_reindex, error_message FROM embedding_status"
        ).fetchall()

    assert temporary_sessions == [(DEMO_CLAUDE_AI_TEMPORARY_SESSION_ID,)]
    assert len(capture_gap_events) == 1
    assert "DOM browser-capture fallback" in capture_gap_events[0][1]
    assert chatgpt_raw_rows == 3
    assert chatgpt_session_rows == 1
    assert ("branch", "prefix-sharing") in links
    assert ("continuation", "spawned-fresh") in links
    assert ("subagent", "spawned-fresh") in links
    assert len(compaction_events) >= 1
    assert sidechain_sessions == [(DEMO_CLAUDE_CODE_LINEAGE_SIDECHAIN_SESSION_ID,)]
    assert subagent_snapshots >= 1
    assert subagent_runs >= 1
    terminal_state_counts = dict(unfinished_terminal_states)
    assert terminal_state_counts.get("question_left", 0) + terminal_state_counts.get("tool_left", 0) >= 1
    assert terminal_state_counts.get("error_left", 0) >= 1
    assert embedding_rows >= 1
    assert embedding_status == [(DEMO_EMBEDDING_PROSE_SESSION_ID, embedding_rows, 0, None)]


@pytest.mark.asyncio
async def test_seed_materializes_session_profiles_for_postmortem(tmp_path: Path) -> None:
    """The no-daemon seed must materialize the session-profile insight read model.

    Without it ``analyze --postmortem`` (and the session-digest surfaces) render
    an empty bundle on the demo archive because the postmortem aggregator fetches
    profiles that ``parse_sources_archive`` never wrote. Guards the #2196 fix.
    """

    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=True)

    with sqlite3.connect(archive_root / "index.db") as conn:
        profile_count = conn.execute("SELECT count(*) FROM session_profiles").fetchone()[0]

    assert profile_count == len(DEMO_SESSION_IDS)


@pytest.mark.asyncio
async def test_demo_verify_reports_missing_overlays(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"

    await seed_demo_archive(archive_root, force=True, with_overlays=False)
    verify = verify_demo_archive(archive_root, require_overlays=True)

    assert verify.ok is False
    assert "expected demo overlays" in "\n".join(verify.problems)
    assert all(row.ok for row in verify.construct_coverage)


@pytest.mark.asyncio
async def test_demo_verify_can_skip_daemon_source_path_leak_posture(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"

    await seed_demo_archive(archive_root, force=True, with_overlays=True)
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET source_path = ?", (str(archive_root / "inbox" / "demo.jsonl"),))

    strict = verify_demo_archive(archive_root, require_overlays=True)
    daemon_wait = verify_demo_archive(
        archive_root,
        require_overlays=True,
        check_source_path_leaks=False,
    )

    assert strict.ok is False
    assert "raw source paths contain absolute paths" in "\n".join(strict.problems)
    assert daemon_wait.ok is True
    assert daemon_wait.absolute_path_leaks == ()


@pytest.mark.asyncio
async def test_seed_injects_demo_cost_for_postmortem(tmp_path: Path) -> None:
    """The demo claude-code session must carry usage so the postmortem blade
    renders real cost + token lanes (not $0) on the demo archive. Guards #2196
    slice 2 and the SessionProfile cost/token round-trip."""

    from polylogue.scenarios import DEMO_CLAUDE_CODE_SESSION_ID

    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=True)

    with sqlite3.connect(archive_root / "index.db") as conn:
        row = conn.execute(
            "SELECT total_cost_usd, total_input_tokens, total_output_tokens FROM session_profiles WHERE session_id = ?",
            (DEMO_CLAUDE_CODE_SESSION_ID,),
        ).fetchone()

    assert row is not None
    total_cost_usd, total_input_tokens, total_output_tokens = row
    assert total_cost_usd > 0
    assert total_input_tokens > 0
    assert total_output_tokens > 0


@pytest.mark.asyncio
async def test_seed_gives_demo_session_canonical_repo(tmp_path: Path) -> None:
    """The demo claude-code session must carry a canonical repo so the
    postmortem `repos_touched` metric renders a project, not an empty list."""

    import json as _json

    from polylogue.scenarios import DEMO_CLAUDE_CODE_SESSION_ID

    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=True)

    with sqlite3.connect(archive_root / "index.db") as conn:
        row = conn.execute(
            "SELECT repo_names_json FROM session_profiles WHERE session_id = ?",
            (DEMO_CLAUDE_CODE_SESSION_ID,),
        ).fetchone()

    assert row is not None
    assert "polylogue" in _json.loads(row[0])
