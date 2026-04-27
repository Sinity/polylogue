"""Tests for typed resume brief construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.session_product_rebuild import rebuild_session_products_sync
from tests.infra.storage_records import ConversationBuilder


def _seed_resume_sessions(db_path: Path) -> None:
    (
        ConversationBuilder(db_path, "resume-root")
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
            text="I will inspect command wiring and product surfaces.",
            timestamp="2026-04-20T10:05:00+00:00",
            provider_meta={
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "tool_name": "Read",
                        "semantic_type": "file_read",
                        "input": {"path": "/workspace/polylogue/polylogue/cli/click_app.py"},
                    }
                ]
            },
        )
        .save()
    )
    (
        ConversationBuilder(db_path, "resume-child")
        .provider("claude-code")
        .title("Resume Continuation")
        .parent_conversation("resume-root")
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
            provider_meta={
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "tool_name": "Bash",
                        "semantic_type": "shell",
                        "input": {"command": "pytest -q tests/unit/core/test_resume.py"},
                    }
                ]
            },
        )
        .save()
    )


@pytest.mark.asyncio
async def test_resume_brief_composes_products_and_related_sessions(cli_workspace: dict[str, Path]) -> None:
    db_path = cli_workspace["db_path"]
    _seed_resume_sessions(db_path)
    with open_connection(db_path) as conn:
        rebuild_session_products_sync(conn)

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    brief = await archive.resume_brief("resume-child")

    assert brief is not None
    assert brief.session_id == "resume-child"
    assert brief.facts.parent_id == "resume-root"
    assert brief.facts.branch_type == "continuation"
    assert brief.facts.message_count == 2
    assert "shell" in brief.facts.tool_categories
    assert brief.inferences.intent_summary == "System crashed, continue the resume command and run focused tests."
    assert brief.inferences.work_events
    assert brief.inferences.work_thread is not None
    assert brief.inferences.work_thread.session_count == 2
    assert any(session.conversation_id == "resume-root" for session in brief.related_sessions)
    assert brief.next_steps
    assert brief.uncertainties == ()


@pytest.mark.asyncio
async def test_resume_brief_degrades_when_products_are_unavailable(cli_workspace: dict[str, Path]) -> None:
    db_path = cli_workspace["db_path"]
    _seed_resume_sessions(db_path)

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    brief = await archive.resume_brief("resume-root")

    assert brief is not None
    assert brief.facts.conversation_id == "resume-root"
    assert brief.facts.message_count == 2
    assert {uncertainty.source for uncertainty in brief.uncertainties} >= {
        "session_profile",
        "session_enrichment",
        "work_thread",
    }
    assert all("session_products" in uncertainty.detail for uncertainty in brief.uncertainties)
