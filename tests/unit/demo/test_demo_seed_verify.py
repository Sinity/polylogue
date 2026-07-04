from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.demo import seed_demo_archive, verify_demo_archive
from polylogue.scenarios import (
    DEMO_CLAUDE_AI_TEMPORARY_SESSION_ID,
    DEMO_CLAUDE_CODE_LINEAGE_SIDECHAIN_SESSION_ID,
    DEMO_CLAUDE_CODE_SESSION_ID,
    DEMO_EMBEDDING_PROSE_SESSION_ID,
    DEMO_SESSION_IDS,
)


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

    with sqlite3.connect(archive_root / "index.db") as conn:
        links = conn.execute("SELECT link_type, inheritance FROM session_links ORDER BY src_session_id").fetchall()
        temporary_sessions = conn.execute("SELECT session_id FROM sessions WHERE session_kind = 'temporary'").fetchall()
        capture_gap_events = conn.execute(
            "SELECT session_id, summary FROM session_events WHERE event_type = 'capture_gap'"
        ).fetchall()
        compaction_events = conn.execute(
            "SELECT session_id, summary FROM session_events WHERE event_type = 'compaction'"
        ).fetchall()
        sidechain_sessions = conn.execute("SELECT session_id FROM sessions WHERE branch_type = 'sidechain'").fetchall()
        subagent_snapshots = conn.execute(
            "SELECT COUNT(*) FROM session_context_snapshots WHERE boundary = 'subagent_start'"
        ).fetchone()[0]
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
    assert ("branch", "prefix-sharing") in links
    assert ("continuation", "spawned-fresh") in links
    assert ("subagent", "spawned-fresh") in links
    assert len(compaction_events) >= 1
    assert sidechain_sessions == [(DEMO_CLAUDE_CODE_LINEAGE_SIDECHAIN_SESSION_ID,)]
    assert subagent_snapshots >= 1
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
