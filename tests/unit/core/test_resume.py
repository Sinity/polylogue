"""Tests for typed resume brief construction (archive ArchiveStore reads)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from polylogue.api import Polylogue
from tests.infra.storage_records import SessionBuilder

# Archive session ids derived from the builder's provider_session_id
# (``ext-<conv_id>``) and the claude-code origin.
ROOT_ID = "claude-code-session:ext-resume-root"
CHILD_ID = "claude-code-session:ext-resume-child"


def _seed_resume_sessions(db_path: Path) -> None:
    (
        SessionBuilder(db_path, "resume-root")
        .provider("claude-code")
        .title("Resume Root")
        .created_at("2026-04-20T10:00:00+00:00")
        .updated_at("2026-04-20T10:15:00+00:00")
        .metadata({"tags": ["issue:309"]})
        .add_message(
            "root-u1",
            role="user",
            text="Implement a resume command for fresh agents.",
            timestamp="2026-04-20T10:00:00+00:00",
        )
        .add_message(
            "root-a1",
            role="assistant",
            text="I will inspect command wiring and insight surfaces.",
            timestamp="2026-04-20T10:05:00+00:00",
            blocks=[
                {
                    "type": "text",
                    "text": "I will inspect command wiring and insight surfaces.",
                },
                {
                    "type": "tool_use",
                    "tool_name": "Read",
                    "semantic_type": "file_read",
                    "input": {"path": "/workspace/polylogue/polylogue/cli/click_app.py"},
                },
            ],
        )
        .save()
    )
    (
        SessionBuilder(db_path, "resume-child")
        .provider("claude-code")
        .title("Resume Continuation")
        .parent_session("ext-resume-root")
        .branch_type("continuation")
        .created_at("2026-04-20T11:00:00+00:00")
        .updated_at("2026-04-20T11:20:00+00:00")
        .add_message(
            "child-u1",
            role="user",
            text="System crashed, continue the resume command and run focused tests.",
            timestamp="2026-04-20T11:00:00+00:00",
        )
        .add_message(
            "child-a1",
            role="assistant",
            text="Continuing implementation and running pytest for resume tests.",
            timestamp="2026-04-20T11:15:00+00:00",
            blocks=[
                {
                    "type": "text",
                    "text": "Continuing implementation and running pytest for resume tests.",
                },
                {
                    "type": "tool_use",
                    "tool_name": "Bash",
                    "semantic_type": "shell",
                    "input": {"command": "pytest -q tests/unit/core/test_resume.py"},
                },
            ],
        )
        .save()
    )


def _set_profile_state(
    db_path: Path,
    session_id: str,
    *,
    terminal_state: str | None = None,
    workflow_shape: str | None = None,
) -> None:
    """Override native session_profiles columns for ranking-signal tests."""
    conn = sqlite3.connect(db_path)
    try:
        assignments: list[str] = []
        params: list[object] = []
        if terminal_state is not None:
            assignments.append("terminal_state = ?")
            params.append(terminal_state)
        if workflow_shape is not None:
            assignments.append("workflow_shape = ?")
            params.append(workflow_shape)
        params.append(session_id)
        conn.execute(
            f"UPDATE session_profiles SET {', '.join(assignments)} WHERE session_id = ?",
            params,
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_resume_brief_composes_insights_and_related_sessions(cli_workspace: dict[str, Path]) -> None:
    db_path = cli_workspace["db_path"]
    _seed_resume_sessions(db_path)

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    await archive.rebuild_insights()
    brief = await archive.resume_brief(CHILD_ID)

    assert brief is not None
    assert brief.session_id == CHILD_ID
    assert brief.facts.parent_id == ROOT_ID
    assert brief.facts.branch_type == "continuation"
    assert brief.facts.message_count == 2
    assert "shell" in brief.facts.tool_categories
    assert brief.inferences.intent_summary == "System crashed, continue the resume command and run focused tests."
    assert brief.inferences.work_events
    assert brief.inferences.thread is not None
    assert brief.inferences.thread.session_count == 2
    assert any(session.session_id == ROOT_ID for session in brief.related_sessions)
    assert brief.next_steps
    assert brief.uncertainties == ()


@pytest.mark.asyncio
async def test_resume_brief_provenance_cites_substrate_rows(cli_workspace: dict[str, Path]) -> None:
    """Provenance must point back at every substrate row composed into the brief."""

    from polylogue.insights.resume import RESUME_BRIEF_MATERIALIZER_VERSION

    db_path = cli_workspace["db_path"]
    _seed_resume_sessions(db_path)

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    await archive.rebuild_insights()
    brief = await archive.resume_brief(CHILD_ID)

    assert brief is not None
    provenance = brief.provenance
    assert provenance.materializer_version == RESUME_BRIEF_MATERIALIZER_VERSION
    assert provenance.computed_at  # ISO timestamp populated
    # Target session is always first cited.
    assert provenance.cited_session_ids[0] == CHILD_ID
    # Related sessions discovered via session-tree / thread must also be cited.
    assert ROOT_ID in provenance.cited_session_ids
    # Every message in the target session must be cited by ID.
    assert provenance.cited_message_ids
    # Work thread, when found, is cited by its substrate ID.
    assert provenance.cited_thread_id is not None


@pytest.mark.asyncio
async def test_resume_brief_degrades_when_insights_are_unavailable(cli_workspace: dict[str, Path]) -> None:
    db_path = cli_workspace["db_path"]
    _seed_resume_sessions(db_path)

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    brief = await archive.resume_brief(ROOT_ID)

    assert brief is not None
    assert brief.facts.session_id == ROOT_ID
    assert brief.facts.message_count == 2
    # Without a rebuild the session_profile (and thus inference) is not
    # materialized, so the brief must flag it as an uncertainty source.
    # Thread topology is durable at write time in the archive
    # store, so it does not degrade — only the profile-derived projections do.
    assert {uncertainty.source for uncertainty in brief.uncertainties} >= {"session_profile"}
    assert all("session_insights" in uncertainty.detail for uncertainty in brief.uncertainties)
    assert brief.next_steps == (
        "Continue from latest assistant state: I will inspect command wiring and insight surfaces.",
    )


@pytest.mark.asyncio
async def test_resume_brief_flags_partial_merged_profile(
    cli_workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = cli_workspace["db_path"]
    _seed_resume_sessions(db_path)

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    await archive.rebuild_insights()
    profile = await archive.get_session_profile_insight(ROOT_ID)
    assert profile is not None
    partial_profile = profile.model_copy(
        update={"evidence": None, "inference": None, "enrichment": None},
    )
    monkeypatch.setattr(
        archive,
        "get_session_profile_insight",
        AsyncMock(return_value=partial_profile),
    )

    brief = await archive.resume_brief(ROOT_ID)

    assert brief is not None
    assert [uncertainty.detail for uncertainty in brief.uncertainties if uncertainty.source == "session_profile"] == [
        "merged session profile is missing: evidence, inference, enrichment",
    ]


@pytest.mark.asyncio
async def test_resume_candidates_rank_and_dedupe_logical_sessions(cli_workspace: dict[str, Path]) -> None:
    db_path = cli_workspace["db_path"]
    _seed_resume_sessions(db_path)
    (
        SessionBuilder(db_path, "clean-chat")
        .provider("claude-code")
        .title("Clean Chat")
        .created_at("2026-04-20T11:00:00+00:00")
        .updated_at("2026-04-20T11:05:00+00:00")
        .add_message("clean-u1", role="user", text="Thanks", timestamp="2026-04-20T11:00:00+00:00")
        .add_message("clean-a1", role="assistant", text="Done.", timestamp="2026-04-20T11:05:00+00:00")
        .save()
    )

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    await archive.rebuild_insights()
    # The candidate surfaces the lineage-strongest terminal_state/workflow_shape
    # across every physical session in the logical group (root + continuation),
    # so the ranking-signal override must cover both members; otherwise the
    # root's computed "tool_left" (weight 1.0) outranks "question_left" (0.85).
    _set_profile_state(db_path, ROOT_ID, terminal_state="question_left", workflow_shape="agentic_loop")
    _set_profile_state(db_path, CHILD_ID, terminal_state="question_left", workflow_shape="agentic_loop")

    candidates = await archive.find_resume_candidates(
        repo_path="/workspace/polylogue",
        cwd="/workspace/polylogue/polylogue/cli",
        recent_files=("/workspace/polylogue/polylogue/cli/click_app.py",),
        limit=10,
    )

    assert candidates
    assert candidates[0].logical_session_id == ROOT_ID
    assert [candidate.logical_session_id for candidate in candidates].count(ROOT_ID) == 1
    assert candidates[0].terminal_state == "question_left"
    assert candidates[0].workflow_shape == "agentic_loop"
    assert candidates[0].file_overlap == ("/workspace/polylogue/polylogue/cli/click_app.py",)
    assert set(candidates[0].score_breakdown) == {
        "recency",
        "file_overlap",
        "cwd_match",
        "terminal_state",
        "workflow_shape",
    }


@pytest.mark.asyncio
async def test_resume_candidates_empty_context_prefers_unfinished_sessions(
    cli_workspace: dict[str, Path],
) -> None:
    db_path = cli_workspace["db_path"]
    _seed_resume_sessions(db_path)

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    await archive.rebuild_insights()
    _set_profile_state(db_path, ROOT_ID, terminal_state="clean_finish")
    _set_profile_state(db_path, CHILD_ID, terminal_state="question_left")

    candidates = await archive.find_resume_candidates(
        repo_path="/workspace/polylogue",
        limit=5,
    )

    assert candidates
    assert all(candidate.terminal_state != "clean_finish" for candidate in candidates)
