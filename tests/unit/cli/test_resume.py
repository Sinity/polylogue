"""Tests for the resume command surface."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
from tests.infra.storage_records import ConversationBuilder


def _seed_resume_session(db_path: Path) -> None:
    (
        ConversationBuilder(db_path, "cli-resume-root")
        .provider("codex")
        .title("CLI Resume")
        .created_at("2026-04-21T09:00:00+00:00")
        .updated_at("2026-04-21T09:30:00+00:00")
        .add_message(
            "u1",
            role="user",
            text="Add resume JSON output.",
            timestamp="2026-04-21T09:00:00+00:00",
        )
        .add_message(
            "a1",
            role="assistant",
            text="Implemented JSON output and ready to verify.",
            timestamp="2026-04-21T09:25:00+00:00",
        )
        .save()
    )
    with open_connection(db_path) as conn:
        rebuild_session_insights_sync(conn)


def test_resume_json_by_root_format(cli_workspace: dict[str, Path]) -> None:
    _seed_resume_session(cli_workspace["db_path"])

    result = CliRunner().invoke(
        cli,
        ["--format", "json", "resume", "cli-resume-root"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    brief = payload["result"]
    assert brief["session_id"] == "cli-resume-root"
    assert brief["facts"]["title"] == "CLI Resume"
    assert set(brief) >= {"facts", "inferences", "related_sessions", "uncertainties", "next_steps"}


def test_resume_plain_names_evidence_sections(cli_workspace: dict[str, Path]) -> None:
    _seed_resume_session(cli_workspace["db_path"])

    result = CliRunner().invoke(cli, ["resume", "cli-resume-root"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "Resume Brief" in result.output
    assert "Facts" in result.output
    assert "Inferred State" in result.output
    assert "Next Steps" in result.output


def test_resume_missing_session_exits_with_clear_message(cli_workspace: dict[str, Path]) -> None:
    result = CliRunner().invoke(cli, ["resume", "missing-session"], catch_exceptions=False)

    assert result.exit_code == 1
    assert "Conversation not found: missing-session" in str(result.exception)
